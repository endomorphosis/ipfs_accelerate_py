"""SupervisorWorldSnapshot@1 — accelerator-owned content-addressed world contract.

This module is a closed immutable schema. It stores reference CIDs and statuses
only. Semantic payloads, prompts, credentials, responses, local paths, and
provider payloads are forbidden. Datasets roots remain semantic-only.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

SCHEMA = "lgswf/supervisor-world-snapshot@1"
INTERFACE = "SupervisorWorldSnapshot@1"

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

EPOCH_FIELDS = ("coordination_epoch", "fencing_epoch")

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

FORBIDDEN_EMBEDDED_FIELDS = frozenset(
    {
        "raw_source",
        "source",
        "prompt",
        "prompts",
        "prompt_body",
        "credential",
        "credentials",
        "model_response",
        "responses",
        "local_path",
        "mutable_path",
        "provider_payload",
        "provider_payloads",
    }
)

OPERATIONAL_IN_DATASETS_FORBIDDEN = frozenset(
    {
        "claim_id",
        "claims",
        "path",
        "paths",
        "worker",
        "workers",
        "prompt",
        "provider_payload",
        "credential",
        "lease_id",
        "fencing_token",
        "attempt_id",
        "duckdb_path",
        "quack_endpoint",
        "mutable_state",
    }
)

COMPONENT_OWNERS = {
    "repository": "ipfs_accelerate_py",
    "repository_tree": "ipfs_accelerate_py",
    "datasets_repository": "ipfs_datasets_py",
    "datasets_semantic_state_root": "ipfs_datasets_py",
    "symbol_root": "ipfs_datasets_py",
    "capsule_index": "ipfs_datasets_py",
    "environment_bindings": "ipfs_datasets_py",
    "contract_root": "ipfs_datasets_py",
    "obligation_set": "ipfs_datasets_py",
    "accepted_plan_root": "ipfs_accelerate_py",
    "plan_revision": "ipfs_accelerate_py",
    "objectives": "ipfs_accelerate_py",
    "goal_population": "ipfs_accelerate_py",
    "subgoal_population": "ipfs_accelerate_py",
    "task_population": "ipfs_accelerate_py",
    "claims": "ipfs_accelerate_py",
    "resource_snapshot": "ipfs_accelerate_py",
    "capability_snapshot": "ipfs_accelerate_py",
    "merge_root": "ipfs_accelerate_py",
    "completion_root": "ipfs_accelerate_py",
    "unresolved_gaps": "ipfs_accelerate_py",
    "policy_root": "ipfs_accelerate_py",
    "event_cursor": "ipfs_accelerate_py",
    "ducklake_projection_health": "ipfs_accelerate_py",
}

ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema",
        "interface",
        "repository_id",
        "components",
        "coordination_epoch",
        "fencing_epoch",
        "snapshot_cid",
    }
)

ALLOWED_COMPONENT_FIELDS = frozenset({"status", "cid", "owner", "evidence_cid"})


class WorldSnapshotContractError(ValueError):
    """Closed SupervisorWorldSnapshot@1 rejected the payload."""


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_sha256_cid(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value[7:]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _reject_floats(value: object, path: str) -> None:
    if isinstance(value, float):
        raise WorldSnapshotContractError(f"float forbidden at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_floats(item, f"{path}[{index}]")


def _reject_forbidden_keys(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_EMBEDDED_FIELDS:
                raise WorldSnapshotContractError(
                    f"forbidden embedded field {key!r} at {path}"
                )
            _reject_forbidden_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}[{index}]")


def field_ownership_table() -> tuple[dict[str, str], ...]:
    rows = [
        {
            "field": "schema",
            "owner": "ipfs_accelerate_py",
            "authority": "contract",
            "kind": "identity",
        },
        {
            "field": "repository_id",
            "owner": "ipfs_accelerate_py",
            "authority": "repository",
            "kind": "identity",
        },
    ]
    for name in REQUIRED_COMPONENTS:
        rows.append(
            {
                "field": name,
                "owner": COMPONENT_OWNERS[name],
                "authority": (
                    "datasets-semantic" if name in DATASETS_COMPONENTS else "operational"
                ),
                "kind": "reference-root",
            }
        )
    for name in EPOCH_FIELDS:
        rows.append(
            {
                "field": name,
                "owner": "ipfs_accelerate_py",
                "authority": "coordination",
                "kind": "epoch",
            }
        )
    rows.append(
        {
            "field": "ducklake_projection_health",
            "owner": "ipfs_accelerate_py",
            "authority": "optional-non-authoritative-projection",
            "kind": "observation",
        }
    )
    return tuple(rows)


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _unfreeze(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(item) for item in value]
    return value


def mutable_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return _unfreeze(snapshot)


def canonical_bytes(snapshot: Mapping[str, Any]) -> bytes:
    payload = {
        key: _unfreeze(snapshot[key])
        for key in sorted(snapshot)
        if key != "snapshot_cid"
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_snapshot_cid(snapshot: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(snapshot)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def parse_world_snapshot(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and freeze a SupervisorWorldSnapshot@1 payload."""

    if not isinstance(payload, Mapping):
        raise WorldSnapshotContractError("snapshot must be a mapping")
    _reject_floats(payload, "snapshot")
    _reject_forbidden_keys(payload, "snapshot")

    unknown = set(payload) - ALLOWED_TOP_LEVEL
    if unknown:
        raise WorldSnapshotContractError(f"unknown fields: {sorted(unknown)}")

    schema = payload.get("schema")
    if schema != SCHEMA:
        raise WorldSnapshotContractError(f"unsupported schema {schema!r}")
    interface = payload.get("interface", INTERFACE)
    if interface != INTERFACE:
        raise WorldSnapshotContractError(f"unsupported interface {interface!r}")

    repository_id = payload.get("repository_id")
    if not isinstance(repository_id, str) or not repository_id:
        raise WorldSnapshotContractError("repository_id is required")

    for epoch_name in EPOCH_FIELDS:
        epoch = payload.get(epoch_name)
        if type(epoch) is not int or epoch < 0:
            raise WorldSnapshotContractError(f"{epoch_name} must be a non-negative int")

    components = payload.get("components")
    if not isinstance(components, Mapping):
        raise WorldSnapshotContractError("components must be a mapping")
    unknown_components = set(components) - set(REQUIRED_COMPONENTS) - set(OPTIONAL_COMPONENTS)
    if unknown_components:
        raise WorldSnapshotContractError(
            f"unknown components: {sorted(unknown_components)}"
        )
    missing = [name for name in REQUIRED_COMPONENTS if name not in components]
    if missing:
        raise WorldSnapshotContractError(f"missing components: {missing}")

    normalized: dict[str, Any] = {}
    for name, component in components.items():
        if not isinstance(component, Mapping):
            raise WorldSnapshotContractError(f"component {name} must be a mapping")
        extra = set(component) - ALLOWED_COMPONENT_FIELDS
        if extra:
            raise WorldSnapshotContractError(
                f"unknown fields on component {name}: {sorted(extra)}"
            )
        status = component.get("status")
        if status not in COMPONENT_STATUSES:
            raise WorldSnapshotContractError(
                f"malformed status on component {name}: {status!r}"
            )
        cid = component.get("cid", "")
        if status == "unavailable":
            if cid not in ("", None):
                raise WorldSnapshotContractError(
                    f"unavailable component {name} must not carry a cid"
                )
            cid = ""
        elif not is_sha256_cid(cid):
            raise WorldSnapshotContractError(f"malformed CID on component {name}")
        owner = component.get("owner") or COMPONENT_OWNERS[name]
        if owner != COMPONENT_OWNERS[name]:
            raise WorldSnapshotContractError(
                f"repository/owner mismatch on component {name}"
            )
        evidence_cid = component.get("evidence_cid", "")
        if evidence_cid and not is_sha256_cid(evidence_cid):
            raise WorldSnapshotContractError(
                f"malformed evidence CID on component {name}"
            )
        if name in DATASETS_COMPONENTS:
            operational = set(component) & OPERATIONAL_IN_DATASETS_FORBIDDEN
            if operational:
                raise WorldSnapshotContractError(
                    f"operational data inside datasets root {name}: {sorted(operational)}"
                )
        record = {"status": status, "cid": cid, "owner": owner}
        if evidence_cid:
            record["evidence_cid"] = evidence_cid
        normalized[name] = record

    tree = normalized["repository_tree"]
    repo = normalized["repository"]
    if (
        tree["status"] == "current"
        and repo["status"] == "current"
        and tree["owner"] != repo["owner"]
    ):
        raise WorldSnapshotContractError("repository mismatch between tree and repository")

    snapshot: dict[str, Any] = {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "repository_id": repository_id,
        "components": normalized,
        "coordination_epoch": payload["coordination_epoch"],
        "fencing_epoch": payload["fencing_epoch"],
    }
    expected_cid = compute_snapshot_cid(snapshot)
    provided = payload.get("snapshot_cid")
    if provided and provided != expected_cid:
        raise WorldSnapshotContractError("snapshot_cid does not match canonical identity")
    snapshot["snapshot_cid"] = expected_cid
    return _freeze(snapshot)


def example_current_snapshot(*, digest_seed: str = "lgswf-010") -> Mapping[str, Any]:
    """Deterministic valid fixture used by focused tests and later writers."""

    def _cid(label: str) -> str:
        return _sha256_text(f"{digest_seed}:{label}")

    components: dict[str, dict[str, str]] = {}
    for name in REQUIRED_COMPONENTS:
        components[name] = {
            "status": "current",
            "cid": _cid(name),
            "owner": COMPONENT_OWNERS[name],
        }
    return parse_world_snapshot(
        {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "repository_id": "ipfs_accelerate_py",
            "components": components,
            "coordination_epoch": 1,
            "fencing_epoch": 1,
        }
    )
