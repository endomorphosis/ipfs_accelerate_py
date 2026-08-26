"""Immutable configured-board capsule gate for EAAEF supervisor births.

The capsule binds the final host admission, the independently qualified Quack
owner, the accepted control-plane archive, and one exact conflict-free task
frontier.  It grants no authority by itself: the runner must additionally hold
the write-sealed control-plane descriptor and verify it at every child birth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
    build_agent_implementation_control_plane_pin,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_authority_registry import (
    EAAEFAuthorityConflict,
    EAAEFAuthorityRegistry,
    EAAEFAuthorityRegistryError,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
    EAAEF_AUTHORITY_REGISTRY_PREFIX,
    EAAEF_BOARD_NAMESPACE,
    EAAEF_MAXIMUM_LANES,
    EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID,
    EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA,
    ExternalAgentBootstrapAdmissionError,
    external_agent_bootstrap_admission_relative_path,
    verify_external_agent_bootstrap_admission,
)

EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-configured-board-launch-capsule-statement@1"
)
EAAEF_CONFIGURED_BOARD_CAPSULE_APPROVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-configured-board-launch-capsule-approval@1"
)
EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-configured-board-launch-capsule@1"
)
EAAEF_CAPSULE_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-configured-board-launch-capsule-verification@1"
)
EAAEF_LIVE_SEAL_CONFIG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-configured-board-live-seal@2"
)
EAAEF_WORKER_NETWORK_DISPATCH_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-worker-network-dispatch-policy@1"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_PIN_FIELDS: Final = frozenset(
    {
        "schema",
        "runner_path",
        "runner_sha256",
        "capsule_root",
        "capsule_id",
        "source_head",
        "source_tree",
        "archive_sha256",
    }
)
_FRONTIER_TASK_FIELDS: Final = frozenset(
    {
        "task_id",
        "task_cid",
        "dependencies",
        "read_scope",
        "write_scope",
        "effect_scope",
        "container_slots",
        "provider_slots",
    }
)
_ACTIVE_PLAN_FIELDS: Final = frozenset(
    {
        "revision_alias",
        "plan_root_cid",
        "population_cid",
        "semantic_state_root",
        "revision",
        "event_cursor",
    }
)
_PLAN_TRANSITION_FIELDS: Final = frozenset(
    {
        "mode",
        "authorization_cid",
        "transition_receipt_cid",
        "state_observation_cid",
        "before_plan_root_cid",
        "after_plan_root_cid",
        "before_plan_revision",
        "after_plan_revision",
        "before_event_cursor",
        "after_event_cursor",
        "before_semantic_root_cid",
        "after_semantic_root_cid",
    }
)
_CONTROL_PLANE_PROMOTION_FIELDS: Final = frozenset(
    {
        "mode",
        "promotion_receipt_cid",
        "promotion_verification_cid",
        "base_owner_qualification_cid",
        "bootstrap_admission_cid",
        "dispatcher_interface",
        "command_fabric_interface",
        "operation_vocabulary_cid",
        "plan_r2_operational_capability_cid",
        "command_fabric_qualification_cid",
        "authorization_policy_cid",
        "store_id",
        "owner_generation",
        "generic_daemon_gateway_admitted",
    }
)
_STATEMENT_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "board_cid",
        "source_head",
        "source_tree",
        "source_generation_cid",
        "configuration_root",
        "materialization_receipt_cid",
        "materialization_store_generation",
        "materialization_database_program_binding_cid",
        "materialization_bootstrap_profile_cid",
        "materialization_operational_profile_cid",
        "population_cid",
        "plan_root_cid",
        "control_projection_root",
        "coordination_projection_root",
        "execution_projection_root",
        "bootstrap_admission_receipt_cid",
        "admission_operator_did",
        "admission_security_reviewer_did",
        "provider_container_qualification_cid",
        "qualified_worker_image_digest",
        "qualified_worker_container_profile_cid",
        "provider_maximum_parallel_workers",
        "provider_maximum_parallel_containers",
        "provider_worker_principal_did",
        "provider_principal_did",
        "provider_task_dispatch_admitted",
        "provider_workload_class",
        "provider_qualification_signer_did",
        "image_qualification_reviewer_did",
        "quack_owner_qualification_cid",
        "quack_owner_verification_cid",
        "quack_qualification_reviewer_did",
        "quack_owner_principal_did",
        "quack_shard_id",
        "quack_epoch",
        "quack_fence",
        "accepted_control_plane_pin",
        "accepted_control_plane_pin_cid",
        "worker_network_authorization_policy",
        "control_plane_promotion",
        "active_plan",
        "plan_transition",
        "satisfied_task_cids",
        "frontier",
        "frontier_cid",
        "authority",
        "issued_at_ms",
        "expires_at_ms",
        "statement_cid",
    }
)
_EXPECTED_CAPSULE_AUTHORITY: Final = {
    "launch_mode": "configured_board_multi_supervisor",
    "maximum_lanes": EAAEF_MAXIMUM_LANES,
    "actual_lane_count": 0,
    "one_fenced_quack_owner": True,
    "direct_duckdb_file_open": False,
    "child_birth_requires_sealed_descriptor": True,
    "restart_requires_reverification": True,
}
_APPROVAL_FIELDS: Final = frozenset(
    {
        "schema",
        "role",
        "identity_did",
        "statement_cid",
        "issued_at_ms",
        "expires_at_ms",
        "signature",
    }
)
_CAPSULE_FIELDS: Final = frozenset(
    {*_STATEMENT_FIELDS, "reviewer_approval", "capsule_cid"}
)
_SIGNED_COMMAND_FABRIC_PROFILE_FIELDS: Final = frozenset(
    {
        "schema",
        "transport_kind",
        "board_namespace",
        "shard_id",
        "ingress_endpoint",
        "ingress_secret_handle",
        "projection_endpoint",
        "projection_secret_handle",
        "store_id",
        "store_generation",
        "schema_revision",
        "owner_qualification_schema",
        "command_envelope_schema",
        "state_command_schema",
        "ingress_relation",
        "ingress_append_only",
        "ingress_accepts_signed_envelopes_only",
        "operational_database_private",
        "operational_tables_remotely_exposed",
        "one_mutable_owner",
        "owner_verifies_signed_envelopes",
        "projection_read_only",
        "projection_append_allowed",
        "atomic_plan_r2_required",
        "direct_file_fallback",
        "failover_policy",
        "child_adapter_status",
    }
)
_WORKER_NETWORK_DISPATCH_POLICY_FIELDS: Final = frozenset(
    {
        "schema",
        "authorization_schema",
        "verifier_interface",
        "artifact_path_authority",
        "artifact_relative_path_template",
        "dynamic_caller_path_allowed",
        "expected_artifact_cid_required",
        "expected_worker_principal_did_required",
        "expected_provider_principal_did_required",
        "control_plane_capsule_binding_required",
        "task_plan_source_worktree_effect_binding_required",
        "container_and_lease_binding_required",
        "create_start_restart_reverification_required",
        "supported_providers",
        "child_propagation_status",
    }
)
_EXPECTED_WORKER_NETWORK_DISPATCH_POLICY: Final = {
    "schema": EAAEF_WORKER_NETWORK_DISPATCH_POLICY_SCHEMA,
    "authorization_schema": (
        "ipfs_accelerate_py/eaaef-worker-network-authorization@1"
    ),
    "verifier_interface": "verify_worker_network_authorization@1",
    "artifact_path_authority": "verified_invocation_profile_dir",
    "artifact_relative_path_template": (
        "network-authorizations/<sha256(invocation_id)>/<provider>.json"
    ),
    "dynamic_caller_path_allowed": False,
    "expected_artifact_cid_required": True,
    "expected_worker_principal_did_required": True,
    "expected_provider_principal_did_required": True,
    "control_plane_capsule_binding_required": True,
    "task_plan_source_worktree_effect_binding_required": True,
    "container_and_lease_binding_required": True,
    "create_start_restart_reverification_required": True,
    "supported_providers": ["codex", "grok"],
    "child_propagation_status": "unavailable_fail_closed",
}


_VERIFIED_LIVE_SEAL_TOKEN = object()


class VerifiedExternalAgentConfiguredBoardLiveSeal(Mapping[str, Any]):
    """Closed, immutable result of a complete live-seal re-verification.

    A plain mapping with a recomputed ``verification_cid`` is not evidence that
    the admission, capsule, Plan R2 receipts, and accepted control-plane pin
    were reopened.  The exact runtime type is therefore minted only by
    :func:`verify_external_agent_configured_board_live_seal` after all of
    those joins have succeeded.
    """

    __slots__ = ("_value",)

    def __init__(self, token: object, value: Mapping[str, Any]) -> None:
        if token is not _VERIFIED_LIVE_SEAL_TOKEN:
            raise TypeError("verified configured-board live seals come from the verifier")
        # Canonical JSON round-tripping detaches the receipt from every caller
        # owned nested collection.  Consumers receive fresh nested values via
        # ``dict(self._value)`` only at explicit process-boundary projections.
        detached = json.loads(_canonical_bytes(dict(value)).decode("ascii"))
        self._value = MappingProxyType(detached)

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


class ExternalAgentConfiguredBoardCapsuleError(RuntimeError):
    """Fail-closed capsule validation/publication error."""


def validate_eaaef_worker_network_dispatch_policy(value: object) -> dict[str, Any]:
    """Validate the stable per-attempt network-authority consumption policy."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _WORKER_NETWORK_DISPATCH_POLICY_FIELDS
        or dict(value) != _EXPECTED_WORKER_NETWORK_DISPATCH_POLICY
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "worker network authorization policy is not fail closed"
        )
    return dict(value)


def validate_eaaef_operational_command_fabric_profile(
    value: object,
    *,
    operational_program: Mapping[str, Any],
    expected_board_namespace: str,
    expected_shard_id: str,
) -> dict[str, Any]:
    """Validate the split signed-command topology without opening an endpoint."""

    if not isinstance(value, Mapping) or set(value) != _SIGNED_COMMAND_FABRIC_PROFILE_FIELDS:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "operational_command_fabric shape is not canonical"
        )
    profile = dict(value)
    if (
        expected_board_namespace != EAAEF_BOARD_NAMESPACE
        or expected_shard_id != EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "operational_command_fabric expected identity is not admitted"
        )
    exact = {
        "schema": EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA,
        "transport_kind": "signed_command_fabric",
        "board_namespace": expected_board_namespace,
        "shard_id": expected_shard_id,
        "owner_qualification_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-quack-owner-qualification@1"
        ),
        "command_envelope_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/state-command@1"
        ),
        "ingress_relation": "command_inbox",
        "ingress_append_only": True,
        "ingress_accepts_signed_envelopes_only": True,
        "operational_database_private": True,
        "operational_tables_remotely_exposed": False,
        "one_mutable_owner": True,
        "owner_verifies_signed_envelopes": True,
        "projection_read_only": True,
        "projection_append_allowed": False,
        "atomic_plan_r2_required": True,
        "direct_file_fallback": False,
        "failover_policy": "fail_closed",
        "child_adapter_status": "implemented_unqualified_fail_closed",
    }
    if any(profile.get(field) != expected for field, expected in exact.items()):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "operational_command_fabric policy is not fail closed"
        )
    ingress = str(profile.get("ingress_endpoint") or "")
    projection = str(profile.get("projection_endpoint") or "")
    if (
        ingress == projection
        or not ingress.startswith("quack:127.0.0.1:")
        or not projection.startswith("quack:127.0.0.1:")
        or not str(profile.get("ingress_secret_handle") or "").startswith(
            "secret-handle:"
        )
        or not str(profile.get("projection_secret_handle") or "").startswith(
            "secret-handle:"
        )
        or profile.get("store_id") != operational_program.get("store_id")
        or profile.get("store_generation")
        != operational_program.get("store_generation")
        or profile.get("schema_revision")
        != operational_program.get("schema_revision")
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "operational_command_fabric identity is inconsistent"
        )
    return profile


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalAgentConfiguredBoardCapsuleError(
                f"duplicate JSON key in live-seal evidence: {key}"
            )
        result[key] = value
    return result


def _read_stable_repo_json(
    root: Path,
    relative_value: object,
    *,
    noun: str,
    authority_root: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Read one immutable logical authority record from platform state."""

    if not isinstance(relative_value, str) or not relative_value:
        raise ExternalAgentConfiguredBoardCapsuleError(f"{noun} path is missing")
    try:
        registry = EAAEFAuthorityRegistry(
            repo_root=root,
            authority_root=authority_root,
        )
        value = registry.read_json(str(relative_value))
    except EAAEFAuthorityRegistryError as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(
            f"{noun} authority registry read failed: {exc}"
        ) from exc
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _mapping(value: object, noun: str) -> Mapping[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise ExternalAgentConfiguredBoardCapsuleError(f"{noun} is not canonical")
    return value


def _validate_frontier(
    frontier: object,
    *,
    satisfied_task_cids: frozenset[str],
) -> list[dict[str, Any]]:
    if (
        not isinstance(frontier, Sequence)
        or isinstance(frontier, (str, bytes, bytearray))
        or not frontier
        or len(frontier) > EAAEF_MAXIMUM_LANES
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "frontier must contain one to five tasks"
        )
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    cids: set[str] = set()
    for raw in frontier:
        if not isinstance(raw, Mapping) or set(raw) != _FRONTIER_TASK_FIELDS:
            raise ExternalAgentConfiguredBoardCapsuleError(
                "frontier task shape is not canonical"
            )
        task_id = str(raw.get("task_id") or "")
        task_cid = str(raw.get("task_cid") or "")
        dependencies = raw.get("dependencies")
        read_scope = raw.get("read_scope")
        write_scope = raw.get("write_scope")
        effect_scope = raw.get("effect_scope")
        scopes = (dependencies, read_scope, write_scope, effect_scope)
        if (
            not task_id.startswith("EAAEF-")
            or not _SHA256.fullmatch(task_cid)
            or task_id in ids
            or task_cid in cids
            or any(
                not isinstance(scope, list)
                or any(not isinstance(item, str) or not item for item in scope)
                for scope in scopes
            )
            or not read_scope
            or not write_scope
            or not effect_scope
            or not _positive_int(raw.get("container_slots"))
            or not _positive_int(raw.get("provider_slots"))
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "frontier task identity/scope/resource is invalid"
            )
        ids.add(task_id)
        cids.add(task_cid)
        normalized.append(
            {
                "task_id": task_id,
                "task_cid": task_cid,
                "dependencies": sorted(set(dependencies)),
                "read_scope": sorted(set(read_scope)),
                "write_scope": sorted(set(write_scope)),
                "effect_scope": sorted(set(effect_scope)),
                "container_slots": int(raw["container_slots"]),
                "provider_slots": int(raw["provider_slots"]),
            }
        )
    for task in normalized:
        unresolved = set(task["dependencies"]) - satisfied_task_cids
        if unresolved:
            raise ExternalAgentConfiguredBoardCapsuleError(
                f"frontier task {task['task_id']} has unresolved dependencies"
            )
    for index, left in enumerate(normalized):
        left_reads = set(left["read_scope"])
        left_writes = set(left["write_scope"])
        left_effects = set(left["effect_scope"])
        for right in normalized[index + 1 :]:
            right_reads = set(right["read_scope"])
            right_writes = set(right["write_scope"])
            right_effects = set(right["effect_scope"])
            if (
                left_writes & (right_reads | right_writes)
                or right_writes & left_reads
                or left_effects & right_effects
            ):
                raise ExternalAgentConfiguredBoardCapsuleError(
                    "frontier contains overlapping read/write/effect scopes"
                )
    return sorted(normalized, key=lambda item: (item["task_id"], item["task_cid"]))


def _validated_pin(value: object) -> AgentImplementationControlPlanePin:
    mapping = _mapping(value, "accepted control-plane pin")
    if set(mapping) != _PIN_FIELDS:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "accepted control-plane pin shape is not canonical"
        )
    try:
        pin = AgentImplementationControlPlanePin(**dict(mapping))
        observed = build_agent_implementation_control_plane_pin(
            runner_path=pin.runner_path,
            capsule_root=pin.capsule_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "accepted control-plane pin does not verify"
        ) from exc
    if observed != pin:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "accepted control-plane pin changed during verification"
        )
    return pin


def prepare_external_agent_configured_board_capsule(
    *,
    bootstrap_admission: Mapping[str, Any],
    quack_owner_qualification: object,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    accepted_control_plane_pin: object,
    configuration_root: str,
    active_plan: Mapping[str, Any],
    satisfied_task_cids: Sequence[str],
    frontier: Sequence[Mapping[str, Any]],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    trusted_quack_reviewer_dids: Sequence[str],
    worker_network_authorization_policy: Mapping[str, Any],
    now_ms: int,
    expires_at_ms: int,
    plan_r2_transition: Mapping[str, Any] | None = None,
    plan_r2_repository: object | None = None,
) -> dict[str, Any]:
    """Prepare the exact unsigned capsule statement after read-only checks."""

    admission = verify_external_agent_bootstrap_admission(
        bootstrap_admission,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.external_agent_control_plane_promotion import (
        verify_external_agent_control_plane_promotion,
        verify_external_agent_quack_owner_qualification_v1,
    )

    # R1 can consume only the historical owner receipt.  Promotion@2 is a
    # post-bootstrap artifact because its signed PlanR2OperationalCapability@1
    # binds this bootstrap admission CID.  Requiring Promotion@2 in the
    # bootstrap receipt would therefore create an impossible CID cycle.
    if plan_r2_transition is None:
        quack_decision = verify_external_agent_quack_owner_qualification_v1(
            qualification_receipt=quack_owner_qualification,
            board=board,
            materialization_receipt=materialization_receipt,
            expected_source_commit=str(admission["source_head"]),
            expected_source_tree=str(admission["source_tree"]),
            trusted_reviewer_dids=trusted_quack_reviewer_dids,
            now_ms=now_ms,
        )
    else:
        quack_decision = verify_external_agent_control_plane_promotion(
            qualification_receipt=quack_owner_qualification,
            board=board,
            materialization_receipt=materialization_receipt,
            expected_source_commit=str(admission["source_head"]),
            expected_source_tree=str(admission["source_tree"]),
            trusted_reviewer_dids=trusted_quack_reviewer_dids,
            trusted_operator_dids=trusted_operator_dids,
            trusted_security_reviewer_dids=trusted_security_reviewer_dids,
            now_ms=now_ms,
        )
    quack = _mapping(
        quack_decision,
        "Quack qualification decision",
    )
    if quack.get("allowed") is not True:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "Quack owner qualification is not admitted: "
            + ", ".join(str(item) for item in quack.get("blockers") or ())
        )
    common_quack_mismatch = (
        quack.get("shard_id") != admission["quack_shard_id"]
        or quack.get("epoch") != admission["quack_epoch"]
        or quack.get("fence") != admission["quack_fence"]
        or quack.get("owner_principal_did")
        != admission["quack_owner_principal_did"]
    )
    if plan_r2_transition is None:
        phase_mismatch = (
            quack.get("receipt_cid")
            != admission["quack_owner_qualification_cid"]
            or quack.get("decision_cid")
            != admission["quack_owner_verification_cid"]
            or quack.get("historical_only") is not True
            or quack.get("promotion_allowed") is not False
        )
    else:
        phase_mismatch = (
            quack.get("promotion_allowed") is not True
            or quack.get("base_owner_qualification_cid")
            != admission["quack_owner_qualification_cid"]
            or quack.get("bootstrap_admission_cid") != admission["receipt_cid"]
        )
    if common_quack_mismatch or phase_mismatch:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "Quack owner or control-plane promotion differs from bootstrap admission"
        )
    if plan_r2_transition is None:
        promotion_binding: dict[str, Any] = {
            "mode": "owner_only_r1",
            "promotion_receipt_cid": "",
            "promotion_verification_cid": "",
            "base_owner_qualification_cid": str(
                admission["quack_owner_qualification_cid"]
            ),
            "bootstrap_admission_cid": "",
            "dispatcher_interface": "",
            "command_fabric_interface": "",
            "operation_vocabulary_cid": "",
            "plan_r2_operational_capability_cid": "",
            "command_fabric_qualification_cid": "",
            "authorization_policy_cid": "",
            "store_id": str(quack.get("store_id") or ""),
            "owner_generation": int(quack.get("owner_generation") or 0),
            "generic_daemon_gateway_admitted": False,
        }
    else:
        promotion_binding = {
            "mode": "plan_r2_dispatcher_v2",
            "promotion_receipt_cid": str(quack["receipt_cid"]),
            "promotion_verification_cid": str(quack["decision_cid"]),
            "base_owner_qualification_cid": str(
                quack["base_owner_qualification_cid"]
            ),
            "bootstrap_admission_cid": str(quack["bootstrap_admission_cid"]),
            "dispatcher_interface": str(quack["dispatcher_interface"]),
            "command_fabric_interface": str(quack["command_fabric_interface"]),
            "operation_vocabulary_cid": str(quack["operation_vocabulary_cid"]),
            "plan_r2_operational_capability_cid": str(
                quack["plan_r2_operational_capability_cid"]
            ),
            "command_fabric_qualification_cid": str(
                quack["command_fabric_qualification_cid"]
            ),
            "authorization_policy_cid": str(
                quack["authorization_policy_cid"]
            ),
            "store_id": str(quack["store_id"]),
            "owner_generation": int(quack["owner_generation"]),
            "generic_daemon_gateway_admitted": False,
        }
    pin = _validated_pin(accepted_control_plane_pin)
    network_policy = validate_eaaef_worker_network_dispatch_policy(
        worker_network_authorization_policy
    )
    if (
        pin.source_head != admission["source_head"]
        or pin.source_tree != admission["source_tree"]
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "accepted control-plane generation differs from bootstrap admission"
        )
    if not _SHA256.fullmatch(configuration_root):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "configured-board configuration root is invalid"
        )
    if (
        not _positive_int(now_ms)
        or not _positive_int(expires_at_ms)
        or now_ms >= expires_at_ms
        or expires_at_ms > int(bootstrap_admission["expires_at_ms"])
        or expires_at_ms > int(quack["expires_at_ms"])
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule lifetime exceeds admitted evidence"
        )
    if (
        set(active_plan) != _ACTIVE_PLAN_FIELDS
        or any(
            not _SHA256.fullmatch(str(active_plan.get(field) or ""))
            for field in ("plan_root_cid", "population_cid", "semantic_state_root")
        )
        or not _positive_int(active_plan.get("revision"))
        or not isinstance(active_plan.get("event_cursor"), str)
        or not active_plan.get("event_cursor")
    ):
        raise ExternalAgentConfiguredBoardCapsuleError("active plan binding is invalid")
    if plan_r2_transition is None:
        if (
            active_plan.get("revision_alias") != "EAAEF-PLAN-R1"
            or active_plan.get("revision") != 1
            or active_plan.get("plan_root_cid")
            != bootstrap_admission["plan_root_cid"]
            or active_plan.get("population_cid")
            != bootstrap_admission["population_cid"]
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "active R1 plan binding is invalid"
            )
        transition_binding = {
            "mode": "bootstrap_r1",
            "authorization_cid": "",
            "transition_receipt_cid": "",
            "state_observation_cid": str(admission["materialization_receipt_cid"]),
            "before_plan_root_cid": str(active_plan["plan_root_cid"]),
            "after_plan_root_cid": str(active_plan["plan_root_cid"]),
            "before_plan_revision": 1,
            "after_plan_revision": 1,
            "before_event_cursor": str(active_plan["event_cursor"]),
            "after_event_cursor": str(active_plan["event_cursor"]),
            "before_semantic_root_cid": str(active_plan["semantic_state_root"]),
            "after_semantic_root_cid": str(active_plan["semantic_state_root"]),
        }
    else:
        if set(plan_r2_transition) != {
            "authorization",
            "transition_receipt",
            "state_observation",
        }:
            raise ExternalAgentConfiguredBoardCapsuleError(
                "Plan R2 transition evidence shape is invalid"
            )
        from ipfs_accelerate_py.agent_supervisor.planning.external_agent_plan_r2 import (
            ExternalAgentPlanR2Error,
            validate_plan_r2_launch_transition,
        )

        try:
            transition = validate_plan_r2_launch_transition(
                repository=plan_r2_repository,
                authorization=plan_r2_transition["authorization"],
                transition_receipt=plan_r2_transition["transition_receipt"],
                state_observation=plan_r2_transition["state_observation"],
                trusted_operator_dids=trusted_operator_dids,
                trusted_security_reviewer_dids=trusted_security_reviewer_dids,
                now_ms=now_ms,
            )
        except ExternalAgentPlanR2Error as exc:
            raise ExternalAgentConfiguredBoardCapsuleError(str(exc)) from exc
        receipt = plan_r2_transition["transition_receipt"]
        authorization = plan_r2_transition["authorization"]
        if (
            active_plan.get("revision_alias") != "EAAEF-PLAN-R2"
            or active_plan.get("plan_root_cid")
            != transition["active_plan_root_cid"]
            or active_plan.get("population_cid")
            != authorization["population_cid"]
            or active_plan.get("semantic_state_root")
            != transition["semantic_root_cid"]
            or active_plan.get("revision") != transition["active_plan_revision"]
            or active_plan.get("event_cursor") != transition["event_cursor"]
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "active Plan R2 binding differs from transition receipt"
            )
        transition_binding = {
            "mode": "atomic_plan_r2",
            "authorization_cid": str(transition["authorization_cid"]),
            "transition_receipt_cid": str(transition["transition_receipt_cid"]),
            "state_observation_cid": str(transition["state_observation_cid"]),
            "before_plan_root_cid": str(receipt["before_plan_root_cid"]),
            "after_plan_root_cid": str(receipt["after_plan_root_cid"]),
            "before_plan_revision": int(receipt["before_plan_revision"]),
            "after_plan_revision": int(receipt["after_plan_revision"]),
            "before_event_cursor": str(receipt["before_event_cursor"]),
            "after_event_cursor": str(receipt["after_event_cursor"]),
            "before_semantic_root_cid": str(receipt["before_semantic_root_cid"]),
            "after_semantic_root_cid": str(receipt["after_semantic_root_cid"]),
        }
    satisfied = frozenset(str(item) for item in satisfied_task_cids)
    if not satisfied or any(not _SHA256.fullmatch(item) for item in satisfied):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "satisfied task population is missing or invalid"
        )
    normalized_frontier = _validate_frontier(frontier, satisfied_task_cids=satisfied)
    if (
        transition_binding["mode"] == "atomic_plan_r2"
        and [item["task_cid"] for item in normalized_frontier]
        != list(plan_r2_transition["transition_receipt"]["frontier_task_cids"])
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "launch frontier differs from the atomic Plan R2 receipt"
        )
    statement: dict[str, Any] = {
        "schema": EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA,
        "board_namespace": str(bootstrap_admission["board_namespace"]),
        "board_cid": str(admission["board_cid"]),
        "source_head": str(admission["source_head"]),
        "source_tree": str(admission["source_tree"]),
        "source_generation_cid": str(bootstrap_admission["source_generation_cid"]),
        "configuration_root": configuration_root,
        "materialization_receipt_cid": str(admission["materialization_receipt_cid"]),
        "materialization_store_generation": str(
            admission["materialization_store_generation"]
        ),
        "materialization_database_program_binding_cid": str(
            admission["materialization_database_program_binding_cid"]
        ),
        "materialization_bootstrap_profile_cid": str(
            admission["materialization_bootstrap_profile_cid"]
        ),
        "materialization_operational_profile_cid": str(
            admission["materialization_operational_profile_cid"]
        ),
        "population_cid": str(bootstrap_admission["population_cid"]),
        "plan_root_cid": str(bootstrap_admission["plan_root_cid"]),
        "control_projection_root": str(bootstrap_admission["control_projection_root"]),
        "coordination_projection_root": str(
            bootstrap_admission["coordination_projection_root"]
        ),
        "execution_projection_root": str(bootstrap_admission["execution_projection_root"]),
        "bootstrap_admission_receipt_cid": str(admission["receipt_cid"]),
        "admission_operator_did": str(admission["operator_identity_did"]),
        "admission_security_reviewer_did": str(
            admission["security_reviewer_identity_did"]
        ),
        "provider_container_qualification_cid": str(
            admission["provider_container_qualification_cid"]
        ),
        "qualified_worker_image_digest": str(admission["image_digest"]),
        "qualified_worker_container_profile_cid": str(
            admission["container_profile_cid"]
        ),
        "provider_maximum_parallel_workers": int(
            admission["provider_maximum_parallel_workers"]
        ),
        "provider_maximum_parallel_containers": int(
            admission["provider_maximum_parallel_containers"]
        ),
        "provider_worker_principal_did": str(
            admission["provider_worker_principal_did"]
        ),
        "provider_principal_did": str(admission["provider_principal_did"]),
        "provider_task_dispatch_admitted": bool(
            admission["provider_task_dispatch_admitted"]
        ),
        "provider_workload_class": str(admission["provider_workload_class"]),
        "provider_qualification_signer_did": str(
            admission["provider_qualification_signer_did"]
        ),
        "image_qualification_reviewer_did": str(
            admission["image_qualification_reviewer_did"]
        ),
        "quack_owner_qualification_cid": str(admission["quack_owner_qualification_cid"]),
        "quack_owner_verification_cid": str(
            admission["quack_owner_verification_cid"]
        ),
        "quack_qualification_reviewer_did": str(
            admission["quack_qualification_reviewer_did"]
        ),
        "quack_owner_principal_did": str(
            admission["quack_owner_principal_did"]
        ),
        "quack_shard_id": str(quack["shard_id"]),
        "quack_epoch": int(quack["epoch"]),
        "quack_fence": int(quack["fence"]),
        "accepted_control_plane_pin": pin.as_dict(),
        "accepted_control_plane_pin_cid": _cid(pin.as_dict()),
        "worker_network_authorization_policy": network_policy,
        "control_plane_promotion": promotion_binding,
        "active_plan": dict(active_plan),
        "plan_transition": transition_binding,
        "satisfied_task_cids": sorted(satisfied),
        "frontier": normalized_frontier,
        "frontier_cid": _cid(
            {
                "schema": "EAAEFConflictFreeFrontier@1",
                "tasks": normalized_frontier,
                "satisfied_task_cids": sorted(satisfied),
            }
        ),
        "authority": {
            **_EXPECTED_CAPSULE_AUTHORITY,
            "actual_lane_count": len(normalized_frontier),
        },
        "issued_at_ms": now_ms,
        "expires_at_ms": expires_at_ms,
    }
    statement["statement_cid"] = _cid(statement)
    return statement


def prepare_external_agent_capsule_approval(
    statement: Mapping[str, Any],
    *,
    identity_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    _validate_statement(statement)
    if (
        not identity_did.startswith("did:key:")
        or not _positive_int(issued_at_ms)
        or not _positive_int(expires_at_ms)
        or issued_at_ms < int(statement["issued_at_ms"])
        or issued_at_ms >= expires_at_ms
        or expires_at_ms > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentConfiguredBoardCapsuleError("capsule approval is invalid")
    return {
        "schema": EAAEF_CONFIGURED_BOARD_CAPSULE_APPROVAL_SCHEMA,
        "role": "independent_control_plane_reviewer",
        "identity_did": identity_did,
        "statement_cid": statement["statement_cid"],
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }


def _validate_statement(statement: Mapping[str, Any]) -> None:
    body = dict(statement)
    identity = str(body.pop("statement_cid", ""))
    if (
        set(statement) != _STATEMENT_FIELDS
        or statement.get("schema") != EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA
        or identity != _cid(body)
        or not _SHA256.fullmatch(str(statement.get("frontier_cid") or ""))
        or not isinstance(statement.get("authority"), Mapping)
        or dict(statement["authority"])
        != {
            **_EXPECTED_CAPSULE_AUTHORITY,
            "actual_lane_count": len(statement.get("frontier") or ()),
        }
        or not _GIT_OBJECT.fullmatch(str(statement.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(statement.get("source_tree") or ""))
        or any(
            not _SHA256.fullmatch(str(statement.get(field) or ""))
            for field in (
                "board_cid",
                "source_generation_cid",
                "configuration_root",
                "materialization_receipt_cid",
                "materialization_database_program_binding_cid",
                "materialization_bootstrap_profile_cid",
                "materialization_operational_profile_cid",
                "population_cid",
                "plan_root_cid",
                "control_projection_root",
                "coordination_projection_root",
                "execution_projection_root",
                "bootstrap_admission_receipt_cid",
                "provider_container_qualification_cid",
                "qualified_worker_image_digest",
                "qualified_worker_container_profile_cid",
                "quack_owner_qualification_cid",
                "quack_owner_verification_cid",
                "accepted_control_plane_pin_cid",
                "frontier_cid",
            )
        )
        or not _positive_int(statement.get("quack_epoch"))
        or not _positive_int(statement.get("quack_fence"))
        or not _positive_int(statement.get("provider_maximum_parallel_workers"))
        or not _positive_int(
            statement.get("provider_maximum_parallel_containers")
        )
        or int(statement.get("provider_maximum_parallel_workers") or 0)
        > int(statement.get("provider_maximum_parallel_containers") or 0)
        or int(statement.get("provider_maximum_parallel_containers") or 0)
        > EAAEF_MAXIMUM_LANES
        or len(statement.get("frontier") or ())
        > int(statement.get("provider_maximum_parallel_workers") or 0)
        or statement.get("provider_task_dispatch_admitted") is not True
        or not str(statement.get("provider_workload_class") or "")
        or not _positive_int(statement.get("issued_at_ms"))
        or not _positive_int(statement.get("expires_at_ms"))
        or int(statement.get("issued_at_ms") or 0)
        >= int(statement.get("expires_at_ms") or 0)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            str(statement.get("materialization_store_generation") or ""),
        )
    ):
        raise ExternalAgentConfiguredBoardCapsuleError("capsule statement is invalid")
    reviewer_fields = (
        "admission_operator_did",
        "admission_security_reviewer_did",
        "provider_qualification_signer_did",
        "image_qualification_reviewer_did",
        "quack_qualification_reviewer_did",
        "provider_worker_principal_did",
        "provider_principal_did",
        "quack_owner_principal_did",
    )
    reviewer_dids = [str(statement.get(field) or "") for field in reviewer_fields]
    if any(not item.startswith("did:key:z") for item in reviewer_dids):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule evidence reviewer identity is invalid"
        )
    if len(
        {
            str(statement["provider_worker_principal_did"]),
            str(statement["provider_principal_did"]),
            str(statement["quack_owner_principal_did"]),
        }
    ) != 3 or set(
        {
            str(statement["provider_worker_principal_did"]),
            str(statement["provider_principal_did"]),
            str(statement["quack_owner_principal_did"]),
        }
    ).intersection(
        {
            str(statement["provider_qualification_signer_did"]),
            str(statement["image_qualification_reviewer_did"]),
            str(statement["quack_qualification_reviewer_did"]),
        }
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule runtime principals are not independent"
        )
    validate_eaaef_worker_network_dispatch_policy(
        statement.get("worker_network_authorization_policy")
    )
    promotion = statement.get("control_plane_promotion")
    if (
        not isinstance(promotion, Mapping)
        or set(promotion) != _CONTROL_PLANE_PROMOTION_FIELDS
        or promotion.get("mode")
        not in {"owner_only_r1", "plan_r2_dispatcher_v2"}
        or promotion.get("base_owner_qualification_cid")
        != statement.get("quack_owner_qualification_cid")
        or not isinstance(promotion.get("store_id"), str)
        or not promotion.get("store_id")
        or not _positive_int(promotion.get("owner_generation"))
        or promotion.get("generic_daemon_gateway_admitted") is not False
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule control-plane promotion binding is invalid"
        )
    promotion_cid_fields = (
        "promotion_receipt_cid",
        "promotion_verification_cid",
        "bootstrap_admission_cid",
        "operation_vocabulary_cid",
        "plan_r2_operational_capability_cid",
        "command_fabric_qualification_cid",
        "authorization_policy_cid",
    )
    if promotion["mode"] == "owner_only_r1":
        if (
            any(promotion.get(field) != "" for field in promotion_cid_fields)
            or promotion.get("dispatcher_interface") != ""
            or promotion.get("command_fabric_interface") != ""
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "R1 capsule cannot claim Plan-R2 dispatcher promotion"
            )
    elif (
        any(
            not _SHA256.fullmatch(str(promotion.get(field) or ""))
            for field in promotion_cid_fields
        )
        or promotion.get("bootstrap_admission_cid")
        != statement.get("bootstrap_admission_receipt_cid")
        or promotion.get("dispatcher_interface")
        != "AuthorizedStateCommandPlanR2OwnerGateway@1"
        or promotion.get("command_fabric_interface") != "QuackCommandFabric@1"
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "Plan-R2 capsule lacks exact Promotion@2 evidence"
        )
    active_plan = statement.get("active_plan")
    if (
        not isinstance(active_plan, Mapping)
        or set(active_plan) != _ACTIVE_PLAN_FIELDS
        or active_plan.get("revision_alias") not in {"EAAEF-PLAN-R1", "EAAEF-PLAN-R2"}
        or not _SHA256.fullmatch(str(active_plan.get("semantic_state_root") or ""))
        or not _SHA256.fullmatch(str(active_plan.get("plan_root_cid") or ""))
        or not _SHA256.fullmatch(str(active_plan.get("population_cid") or ""))
        or not _positive_int(active_plan.get("revision"))
        or not isinstance(active_plan.get("event_cursor"), str)
        or not active_plan.get("event_cursor")
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule active plan is invalid"
        )
    transition = statement.get("plan_transition")
    if not isinstance(transition, Mapping) or set(transition) != _PLAN_TRANSITION_FIELDS:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule plan transition binding is invalid"
        )
    mode = transition.get("mode")
    if (mode == "bootstrap_r1") != (
        promotion["mode"] == "owner_only_r1"
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "plan transition and control-plane promotion phases differ"
        )
    common_invalid = (
        transition.get("after_plan_root_cid") != active_plan.get("plan_root_cid")
        or transition.get("after_plan_revision") != active_plan.get("revision")
        or transition.get("after_event_cursor") != active_plan.get("event_cursor")
        or transition.get("after_semantic_root_cid")
        != active_plan.get("semantic_state_root")
        or any(
            not _SHA256.fullmatch(str(transition.get(field) or ""))
            for field in (
                "state_observation_cid",
                "before_plan_root_cid",
                "after_plan_root_cid",
                "before_semantic_root_cid",
                "after_semantic_root_cid",
            )
        )
        or not _positive_int(transition.get("before_plan_revision"))
        or not _positive_int(transition.get("after_plan_revision"))
        or not isinstance(transition.get("before_event_cursor"), str)
        or not transition.get("before_event_cursor")
        or not isinstance(transition.get("after_event_cursor"), str)
        or not transition.get("after_event_cursor")
    )
    if common_invalid:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule plan transition binding is invalid"
        )
    if mode == "bootstrap_r1":
        if (
            active_plan.get("revision_alias") != "EAAEF-PLAN-R1"
            or active_plan.get("revision") != 1
            or active_plan.get("plan_root_cid") != statement.get("plan_root_cid")
            or active_plan.get("population_cid") != statement.get("population_cid")
            or transition.get("authorization_cid") != ""
            or transition.get("transition_receipt_cid") != ""
            or transition.get("before_plan_root_cid")
            != transition.get("after_plan_root_cid")
            or transition.get("before_plan_revision") != 1
            or transition.get("after_plan_revision") != 1
            or transition.get("before_event_cursor")
            != transition.get("after_event_cursor")
            or transition.get("before_semantic_root_cid")
            != transition.get("after_semantic_root_cid")
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "capsule R1 bootstrap transition is invalid"
            )
    elif mode == "atomic_plan_r2":
        if (
            active_plan.get("revision_alias") != "EAAEF-PLAN-R2"
            or int(active_plan.get("revision") or 0) < 2
            or not _SHA256.fullmatch(str(transition.get("authorization_cid") or ""))
            or not _SHA256.fullmatch(
                str(transition.get("transition_receipt_cid") or "")
            )
            or int(transition.get("after_plan_revision") or 0)
            <= int(transition.get("before_plan_revision") or 0)
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "capsule atomic Plan R2 transition is invalid"
            )
    else:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule plan transition mode is unsupported"
        )
    satisfied = statement.get("satisfied_task_cids")
    if (
        not isinstance(satisfied, list)
        or not satisfied
        or satisfied != sorted(set(satisfied))
        or any(not _SHA256.fullmatch(str(item)) for item in satisfied)
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule satisfied task population is invalid"
        )
    frontier = _validate_frontier(
        statement.get("frontier"),
        satisfied_task_cids=frozenset(statement.get("satisfied_task_cids") or ()),
    )
    if frontier != statement.get("frontier"):
        raise ExternalAgentConfiguredBoardCapsuleError("capsule frontier is noncanonical")
    expected_frontier_cid = _cid(
        {
            "schema": "EAAEFConflictFreeFrontier@1",
            "tasks": frontier,
            "satisfied_task_cids": sorted(statement["satisfied_task_cids"]),
        }
    )
    if expected_frontier_cid != statement["frontier_cid"]:
        raise ExternalAgentConfiguredBoardCapsuleError("capsule frontier identity is invalid")


def assemble_external_agent_configured_board_capsule(
    statement: Mapping[str, Any],
    *,
    reviewer_approval: Mapping[str, Any],
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    _validate_statement(statement)
    reviewer = _verify_capsule_approval(
        reviewer_approval,
        statement=statement,
        trusted_reviewer_dids=trusted_reviewer_dids,
        now_ms=now_ms,
    )
    _require_independent_capsule_reviewer(reviewer, statement=statement)
    capsule = {
        **dict(statement),
        "schema": EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA,
        "reviewer_approval": dict(reviewer_approval),
    }
    capsule["capsule_cid"] = _cid(capsule)
    return capsule


def _verify_capsule_approval(
    approval: object,
    *,
    statement: Mapping[str, Any],
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> str:
    if not isinstance(approval, Mapping) or set(approval) != _APPROVAL_FIELDS:
        raise ExternalAgentConfiguredBoardCapsuleError("capsule approval is invalid")
    identity = str(approval.get("identity_did") or "")
    if (
        approval.get("schema") != EAAEF_CONFIGURED_BOARD_CAPSULE_APPROVAL_SCHEMA
        or approval.get("role") != "independent_control_plane_reviewer"
        or identity not in frozenset(trusted_reviewer_dids)
        or approval.get("statement_cid") != statement.get("statement_cid")
        or not _positive_int(approval.get("issued_at_ms"))
        or not _positive_int(approval.get("expires_at_ms"))
        or int(approval["issued_at_ms"]) > now_ms
        or now_ms >= int(approval["expires_at_ms"])
        or int(approval["expires_at_ms"]) > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentConfiguredBoardCapsuleError("capsule approval is invalid")
    payload = dict(approval)
    signature = payload.pop("signature", None)
    if not isinstance(signature, str) or not signature:
        raise ExternalAgentConfiguredBoardCapsuleError("capsule approval is unsigned")
    try:
        verify_did_key_signature(
            identity_did=identity,
            payload=payload,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule approval signature is invalid"
        ) from exc
    return identity


def _require_independent_capsule_reviewer(
    reviewer: str,
    *,
    statement: Mapping[str, Any],
) -> None:
    """Apply the same role-separation rule to assembly and direct verification."""

    if reviewer in {
        statement["admission_operator_did"],
        statement["provider_worker_principal_did"],
        statement["provider_principal_did"],
        statement["quack_owner_principal_did"],
    }:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "control-plane reviewer must be independent of execution principals"
        )


def verify_external_agent_configured_board_capsule(
    capsule: object,
    *,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Verify immutable bytes and source closure; never opens mutable state."""

    if not isinstance(capsule, Mapping) or set(capsule) != _CAPSULE_FIELDS:
        raise ExternalAgentConfiguredBoardCapsuleError("configured-board capsule is invalid")
    body = dict(capsule)
    capsule_cid = str(body.pop("capsule_cid", ""))
    approval = body.pop("reviewer_approval", None)
    if capsule_cid != _cid({**body, "reviewer_approval": approval}):
        raise ExternalAgentConfiguredBoardCapsuleError("capsule self-address is invalid")
    statement = dict(body)
    statement["schema"] = EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA
    _validate_statement(statement)
    reviewer = _verify_capsule_approval(
        approval,
        statement=statement,
        trusted_reviewer_dids=trusted_reviewer_dids,
        now_ms=now_ms,
    )
    _require_independent_capsule_reviewer(reviewer, statement=statement)
    if now_ms >= int(statement["expires_at_ms"]):
        raise ExternalAgentConfiguredBoardCapsuleError("configured-board capsule expired")
    pin = _validated_pin(statement["accepted_control_plane_pin"])
    if _cid(pin.as_dict()) != statement["accepted_control_plane_pin_cid"]:
        raise ExternalAgentConfiguredBoardCapsuleError("capsule pin identity is invalid")
    if (
        pin.source_head != statement["source_head"]
        or pin.source_tree != statement["source_tree"]
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "capsule pin source identity is invalid"
        )
    report = {
        "schema": EAAEF_CAPSULE_VERIFICATION_SCHEMA,
        "valid": True,
        "capsule_cid": capsule_cid,
        "board_cid": statement["board_cid"],
        "source_head": statement["source_head"],
        "source_tree": statement["source_tree"],
        "configuration_root": statement["configuration_root"],
        "bootstrap_admission_receipt_cid": statement[
            "bootstrap_admission_receipt_cid"
        ],
        "materialization_store_generation": statement[
            "materialization_store_generation"
        ],
        "materialization_database_program_binding_cid": statement[
            "materialization_database_program_binding_cid"
        ],
        "materialization_bootstrap_profile_cid": statement[
            "materialization_bootstrap_profile_cid"
        ],
        "materialization_operational_profile_cid": statement[
            "materialization_operational_profile_cid"
        ],
        "quack_owner_qualification_cid": statement[
            "quack_owner_qualification_cid"
        ],
        "accepted_control_plane_pin_cid": statement[
            "accepted_control_plane_pin_cid"
        ],
        "frontier_cid": statement["frontier_cid"],
        "frontier_task_cids": [item["task_cid"] for item in statement["frontier"]],
        "active_plan": dict(statement["active_plan"]),
        "plan_transition": dict(statement["plan_transition"]),
        "control_plane_promotion": dict(statement["control_plane_promotion"]),
        "worker_network_authorization_policy": dict(
            statement["worker_network_authorization_policy"]
        ),
        "maximum_lanes": EAAEF_MAXIMUM_LANES,
        "provider_maximum_parallel_workers": statement[
            "provider_maximum_parallel_workers"
        ],
        "qualified_worker_image_digest": statement[
            "qualified_worker_image_digest"
        ],
        "qualified_worker_container_profile_cid": statement[
            "qualified_worker_container_profile_cid"
        ],
        "provider_maximum_parallel_containers": statement[
            "provider_maximum_parallel_containers"
        ],
        "actual_lane_count": len(statement["frontier"]),
        "reviewer_identity_did": reviewer,
        "authority_mutated": False,
        "process_started": False,
    }
    report["verification_cid"] = _cid(report)
    return report


def external_agent_configured_board_launch_capsule_relative_path(
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    """Return one source/plan-addressed create-once launch-capsule path."""

    if registry_prefix != EAAEF_AUTHORITY_REGISTRY_PREFIX:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "authority registry prefix is not the reviewed EAAEF prefix"
        )
    if not _GIT_OBJECT.fullmatch(str(source_head or "")) or not _SHA256.fullmatch(
        str(plan_root_cid or "")
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "launch capsule source/plan identity is invalid"
        )
    return Path(registry_prefix) / (
        "configured-board-launch-capsule--"
        f"{source_head}--{plan_root_cid.removeprefix('sha256:')}.json"
    )


def verify_external_agent_configured_board_live_seal(
    live_seal_config: object,
    *,
    repo_root: str | Path,
    configuration_root: str,
    expected_source_head: str,
    expected_source_tree: str,
    accepted_control_plane_pin: AgentImplementationControlPlanePin,
    now_ms: int,
    expected_active_plan_root_cid: str = "",
    plan_r2_repository: object | None = None,
    authority_root: str | Path | None = None,
) -> VerifiedExternalAgentConfiguredBoardLiveSeal:
    """Re-open and join post-freeze evidence at a process-birth boundary.

    The tracked configuration intentionally contains paths and trust anchors,
    not the post-freeze receipt CIDs.  This avoids a source/CID fixed-point
    cycle.  The observed self-addresses are returned for the caller's lifecycle
    identity and must be re-observed on every restart.
    """

    expected_fields = {
        "schema",
        "authority_registry_prefix",
        "bootstrap_admission_schema",
        "configured_board_launch_capsule_schema",
        "trusted_operator_dids",
        "trusted_security_reviewer_dids",
        "trusted_capsule_reviewer_dids",
        "worker_network_authorization_policy",
        "maximum_lanes",
    }
    if not isinstance(live_seal_config, Mapping) or set(live_seal_config) != expected_fields:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "configured_board_live_seal shape is not canonical"
        )
    if (
        live_seal_config.get("schema") != EAAEF_LIVE_SEAL_CONFIG_SCHEMA
        or live_seal_config.get("authority_registry_prefix")
        != EAAEF_AUTHORITY_REGISTRY_PREFIX
        or live_seal_config.get("bootstrap_admission_schema")
        != "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission@1"
        or live_seal_config.get("configured_board_launch_capsule_schema")
        != EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA
        or live_seal_config.get("maximum_lanes") != EAAEF_MAXIMUM_LANES
        or not _SHA256.fullmatch(configuration_root)
        or not _GIT_OBJECT.fullmatch(expected_source_head)
        or not _GIT_OBJECT.fullmatch(expected_source_tree)
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "configured_board_live_seal binding is invalid"
        )
    trust_fields = (
        "trusted_operator_dids",
        "trusted_security_reviewer_dids",
        "trusted_capsule_reviewer_dids",
    )
    trusted: dict[str, tuple[str, ...]] = {}
    for field in trust_fields:
        values = live_seal_config.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item.startswith("did:key:") for item in values)
            or len(values) != len(set(values))
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                f"configured_board_live_seal.{field} is invalid"
            )
        trusted[field] = tuple(values)
    network_policy = validate_eaaef_worker_network_dispatch_policy(
        live_seal_config.get("worker_network_authorization_policy")
    )
    root = Path(repo_root)
    admission_path = external_agent_bootstrap_admission_relative_path(
        expected_source_head,
        registry_prefix=str(live_seal_config["authority_registry_prefix"]),
    )
    admission_receipt, admission_file_sha256 = _read_stable_repo_json(
        root,
        admission_path.as_posix(),
        noun="bootstrap admission receipt",
        authority_root=authority_root,
    )
    try:
        admission = verify_external_agent_bootstrap_admission(
            admission_receipt,
            trusted_operator_dids=trusted["trusted_operator_dids"],
            trusted_security_reviewer_dids=trusted[
                "trusted_security_reviewer_dids"
            ],
            now_ms=now_ms,
        )
    except ExternalAgentBootstrapAdmissionError as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(str(exc)) from exc
    active_plan_root = expected_active_plan_root_cid or str(
        admission["plan_root_cid"]
    )
    capsule_path = external_agent_configured_board_launch_capsule_relative_path(
        expected_source_head,
        active_plan_root,
        registry_prefix=str(live_seal_config["authority_registry_prefix"]),
    )
    capsule_receipt, capsule_file_sha256 = _read_stable_repo_json(
        root,
        capsule_path.as_posix(),
        noun="configured-board launch capsule",
        authority_root=authority_root,
    )
    capsule = verify_external_agent_configured_board_capsule(
        capsule_receipt,
        trusted_reviewer_dids=trusted["trusted_capsule_reviewer_dids"],
        now_ms=now_ms,
    )
    transition_evidence: dict[str, Any] = {
        "authorization_path": "",
        "authorization_file_sha256": "",
        "transition_receipt_path": "",
        "transition_receipt_file_sha256": "",
        "state_observation_path": "",
        "state_observation_file_sha256": "",
        "verification": {},
    }
    if capsule["plan_transition"].get("mode") == "atomic_plan_r2":
        from ipfs_accelerate_py.agent_supervisor.planning.external_agent_plan_r2 import (
            ExternalAgentPlanR2Error,
            plan_r2_state_observation_relative_path,
            plan_r2_transition_authorization_relative_path,
            plan_r2_transition_receipt_relative_path,
            validate_plan_r2_launch_transition,
        )

        plan_root = str(capsule["active_plan"]["plan_root_cid"])
        authorization_path = plan_r2_transition_authorization_relative_path(
            expected_source_head,
            plan_root,
            registry_prefix=str(live_seal_config["authority_registry_prefix"]),
        )
        transition_receipt_path = plan_r2_transition_receipt_relative_path(
            expected_source_head,
            plan_root,
            registry_prefix=str(live_seal_config["authority_registry_prefix"]),
        )
        state_observation_path = plan_r2_state_observation_relative_path(
            expected_source_head,
            plan_root,
            registry_prefix=str(live_seal_config["authority_registry_prefix"]),
        )
        authorization, authorization_file_sha256 = _read_stable_repo_json(
            root,
            authorization_path.as_posix(),
            noun="Plan R2 transition authorization",
            authority_root=authority_root,
        )
        transition_receipt, transition_receipt_file_sha256 = _read_stable_repo_json(
            root,
            transition_receipt_path.as_posix(),
            noun="Plan R2 transition receipt",
            authority_root=authority_root,
        )
        state_observation, state_observation_file_sha256 = _read_stable_repo_json(
            root,
            state_observation_path.as_posix(),
            noun="Plan R2 state observation",
            authority_root=authority_root,
        )
        try:
            transition_verification = validate_plan_r2_launch_transition(
                repository=plan_r2_repository,
                authorization=authorization,
                transition_receipt=transition_receipt,
                state_observation=state_observation,
                trusted_operator_dids=trusted["trusted_operator_dids"],
                trusted_security_reviewer_dids=trusted[
                    "trusted_security_reviewer_dids"
                ],
                now_ms=now_ms,
            )
        except ExternalAgentPlanR2Error as exc:
            raise ExternalAgentConfiguredBoardCapsuleError(str(exc)) from exc
        if (
            transition_verification["authorization_cid"]
            != capsule["plan_transition"]["authorization_cid"]
            or transition_verification["transition_receipt_cid"]
            != capsule["plan_transition"]["transition_receipt_cid"]
            or transition_verification["state_observation_cid"]
            != capsule["plan_transition"]["state_observation_cid"]
            or transition_verification["active_plan_root_cid"]
            != capsule["active_plan"]["plan_root_cid"]
            or transition_verification["active_plan_revision"]
            != capsule["active_plan"]["revision"]
            or transition_verification["event_cursor"]
            != capsule["active_plan"]["event_cursor"]
            or transition_verification["frontier_task_cids"]
            != capsule["frontier_task_cids"]
            or authorization.get("quack_command_fabric_qualification_cid")
            != capsule["control_plane_promotion"][
                "command_fabric_qualification_cid"
            ]
            or transition_receipt.get("capability_cid")
            != capsule["control_plane_promotion"][
                "plan_r2_operational_capability_cid"
            ]
        ):
            raise ExternalAgentConfiguredBoardCapsuleError(
                "Plan R2 birth evidence differs from the launch capsule"
            )
        transition_evidence = {
            "authorization_path": authorization_path.as_posix(),
            "authorization_file_sha256": authorization_file_sha256,
            "transition_receipt_path": transition_receipt_path.as_posix(),
            "transition_receipt_file_sha256": transition_receipt_file_sha256,
            "state_observation_path": state_observation_path.as_posix(),
            "state_observation_file_sha256": state_observation_file_sha256,
            "verification": transition_verification,
        }
    supplied_pin = _validated_pin(accepted_control_plane_pin)
    if (
        admission["source_head"] != expected_source_head
        or admission["source_tree"] != expected_source_tree
        or capsule["source_head"] != expected_source_head
        or capsule["source_tree"] != expected_source_tree
        or capsule["configuration_root"] != configuration_root
        or capsule["bootstrap_admission_receipt_cid"] != admission["receipt_cid"]
        or capsule_receipt.get("board_cid") != admission["board_cid"]
        or capsule_receipt.get("materialization_receipt_cid")
        != admission["materialization_receipt_cid"]
        or capsule_receipt.get("materialization_store_generation")
        != admission["materialization_store_generation"]
        or capsule_receipt.get("materialization_database_program_binding_cid")
        != admission["materialization_database_program_binding_cid"]
        or capsule_receipt.get("materialization_bootstrap_profile_cid")
        != admission["materialization_bootstrap_profile_cid"]
        or capsule_receipt.get("materialization_operational_profile_cid")
        != admission["materialization_operational_profile_cid"]
        or capsule_receipt.get("population_cid") != admission["population_cid"]
        or capsule_receipt.get("plan_root_cid") != admission["plan_root_cid"]
        or capsule_receipt.get("control_projection_root")
        != admission["control_projection_root"]
        or capsule_receipt.get("coordination_projection_root")
        != admission["coordination_projection_root"]
        or capsule_receipt.get("execution_projection_root")
        != admission["execution_projection_root"]
        or capsule_receipt.get("provider_container_qualification_cid")
        != admission["provider_container_qualification_cid"]
        or capsule_receipt.get("qualified_worker_image_digest")
        != admission["image_digest"]
        or capsule_receipt.get("qualified_worker_container_profile_cid")
        != admission["container_profile_cid"]
        or capsule_receipt.get("provider_maximum_parallel_workers")
        != admission["provider_maximum_parallel_workers"]
        or capsule_receipt.get("provider_maximum_parallel_containers")
        != admission["provider_maximum_parallel_containers"]
        or capsule_receipt.get("provider_worker_principal_did")
        != admission["provider_worker_principal_did"]
        or capsule_receipt.get("provider_principal_did")
        != admission["provider_principal_did"]
        or capsule_receipt.get("provider_task_dispatch_admitted")
        != admission["provider_task_dispatch_admitted"]
        or capsule_receipt.get("provider_workload_class")
        != admission["provider_workload_class"]
        or capsule_receipt.get("quack_owner_qualification_cid")
        != admission["quack_owner_qualification_cid"]
        or capsule_receipt.get("quack_owner_verification_cid")
        != admission["quack_owner_verification_cid"]
        or capsule_receipt.get("quack_shard_id") != admission["quack_shard_id"]
        or capsule_receipt.get("quack_epoch") != admission["quack_epoch"]
        or capsule_receipt.get("quack_fence") != admission["quack_fence"]
        or capsule_receipt.get("quack_owner_principal_did")
        != admission["quack_owner_principal_did"]
        or capsule["control_plane_promotion"].get(
            "base_owner_qualification_cid"
        )
        != admission["quack_owner_qualification_cid"]
        or capsule["accepted_control_plane_pin_cid"] != _cid(supplied_pin.as_dict())
        or capsule_receipt.get("accepted_control_plane_pin") != supplied_pin.as_dict()
        or capsule["maximum_lanes"] != EAAEF_MAXIMUM_LANES
        or capsule["active_plan"].get("plan_root_cid") != active_plan_root
        or capsule.get("worker_network_authorization_policy") != network_policy
    ):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "configured-board live-seal evidence has crossed identities"
        )
    report = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-configured-board-live-seal-verification@1"
        ),
        "valid": True,
        "configuration_root": configuration_root,
        "source_head": expected_source_head,
        "source_tree": expected_source_tree,
        "bootstrap_admission_receipt_cid": admission["receipt_cid"],
        "configured_board_capsule_cid": capsule["capsule_cid"],
        "bootstrap_admission_file_sha256": admission_file_sha256,
        "configured_board_capsule_file_sha256": capsule_file_sha256,
        "bootstrap_admission_path": admission_path.as_posix(),
        "configured_board_launch_capsule_path": capsule_path.as_posix(),
        "accepted_control_plane_pin_cid": capsule[
            "accepted_control_plane_pin_cid"
        ],
        "frontier_cid": capsule["frontier_cid"],
        "frontier_task_cids": list(capsule["frontier_task_cids"]),
        "active_plan": dict(capsule["active_plan"]),
        "plan_transition": dict(capsule["plan_transition"]),
        "control_plane_promotion": dict(capsule["control_plane_promotion"]),
        "plan_transition_evidence": transition_evidence,
        "provider_worker_principal_did": admission[
            "provider_worker_principal_did"
        ],
        "provider_principal_did": admission["provider_principal_did"],
        "qualified_worker_image_digest": admission["image_digest"],
        "qualified_worker_container_profile_cid": admission[
            "container_profile_cid"
        ],
        "worker_network_authorization_policy": network_policy,
        "maximum_lanes": EAAEF_MAXIMUM_LANES,
        "actual_lane_count": capsule["actual_lane_count"],
        "authority_mutated": False,
        "process_started": False,
    }
    report["verification_cid"] = _cid(report)
    return VerifiedExternalAgentConfiguredBoardLiveSeal(
        _VERIFIED_LIVE_SEAL_TOKEN,
        report,
    )


def publish_external_agent_configured_board_capsule(
    repo_root: str | Path,
    capsule: Mapping[str, Any],
    *,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
    authority_root: str | Path | None = None,
) -> dict[str, Any]:
    verification = verify_external_agent_configured_board_capsule(
        capsule,
        trusted_reviewer_dids=trusted_reviewer_dids,
        now_ms=now_ms,
    )
    active_plan = verification.get("active_plan")
    if not isinstance(active_plan, Mapping):
        raise ExternalAgentConfiguredBoardCapsuleError(
            "launch capsule active plan is missing"
        )
    relative_path = external_agent_configured_board_launch_capsule_relative_path(
        str(verification["source_head"]),
        str(active_plan.get("plan_root_cid") or ""),
    )
    try:
        registry = EAAEFAuthorityRegistry(
            repo_root=repo_root,
            authority_root=authority_root,
        )
        registry.publish_json(relative_path, capsule)
    except EAAEFAuthorityConflict as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(
            "refusing to overwrite immutable configured-board capsule with "
            "different bytes"
        ) from exc
    except EAAEFAuthorityRegistryError as exc:
        raise ExternalAgentConfiguredBoardCapsuleError(
            f"configured-board capsule registry rejected publication: {exc}"
        ) from exc
    return verification


__all__ = (
    "EAAEF_CAPSULE_VERIFICATION_SCHEMA",
    "EAAEF_CONFIGURED_BOARD_CAPSULE_APPROVAL_SCHEMA",
    "EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA",
    "EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA",
    "EAAEF_LIVE_SEAL_CONFIG_SCHEMA",
    "EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID",
    "EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA",
    "EAAEF_WORKER_NETWORK_DISPATCH_POLICY_SCHEMA",
    "ExternalAgentConfiguredBoardCapsuleError",
    "VerifiedExternalAgentConfiguredBoardLiveSeal",
    "assemble_external_agent_configured_board_capsule",
    "prepare_external_agent_capsule_approval",
    "prepare_external_agent_configured_board_capsule",
    "external_agent_configured_board_launch_capsule_relative_path",
    "publish_external_agent_configured_board_capsule",
    "validate_eaaef_operational_command_fabric_profile",
    "validate_eaaef_worker_network_dispatch_policy",
    "verify_external_agent_configured_board_capsule",
    "verify_external_agent_configured_board_live_seal",
)
