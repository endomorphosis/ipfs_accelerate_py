"""Fail-closed EAAEF container proposal dispatch.

This module is the execution boundary between a database-authoritative task
attempt and an isolated OCI worker.  It intentionally does *not* reuse
``DatabasePortalExecutionBridge``: Portal may run a provider, mutate a
worktree, and merge before the outer database daemon records its idempotency
receipt.

An :class:`ExternalAgentContainerWorkerDispatcher` instead requires one exact
provider/effect claim to be durably reserved through the Quack owner before a
launcher is called.  A worker can only return a patch proposal and artifact
references.  Independent verification and a separately observed host merge
admission are required before the database daemon may report validation.

The currently reviewed 39-operation Quack adapter marks its provider/effect
operations unqualified.  Consequently no production builder installs this
dispatcher yet.  The state machine remains useful and testable, but fails
closed whenever the owner returns a typed no-go or an unrecognised receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ..runtime.worker_network_dispatch import EAAEF_BOARD_NAMESPACE
from ..task_sources.eaaef_execution_contracts import (
    EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA,
)

EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE: Final = (
    "ExternalAgentContainerWorkerDispatcher@1"
)
EXTERNAL_AGENT_CONTAINER_WORK_PACKET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-container-work-packet@1"
)
EXTERNAL_AGENT_CONTAINER_DISPATCH_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-container-dispatch-claim@1"
)
EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-container-dispatch-reservation@1"
)
EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-container-proposal-receipt@1"
)
EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-container-verification-receipt@1"
)
EXTERNAL_AGENT_CONTAINER_ACCEPTED_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-container-accepted-result@1"
)
EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-host-merge-admission@1"
)
EXTERNAL_AGENT_CONTAINER_EFFECT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-container-effect-receipt@1"
)

# These are current qualification facts, not runtime feature flags.  A caller
# cannot change them through a task, transcript, environment variable, or
# repository file.
EXTERNAL_AGENT_CONTAINER_DISPATCH_STATUS: Final = "unavailable_fail_closed"
EXTERNAL_AGENT_CONTAINER_DISPATCH_BLOCKERS: Final = (
    "provider_reservation_before_container_launch_unqualified",
    "provider_receipt_independent_verification_unqualified",
    "effect_reservation_before_external_effect_unqualified",
    "effect_receipt_independent_verification_unqualified",
    "canonical_validation_schema_transaction_adapter_unqualified",
    "source_addressed_container_execution_profile_launch_unqualified",
)

_CID = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_TREE = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})")
_BOUNDED_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}")
_PROVIDERS = frozenset({"codex", "grok"})


class ExternalAgentContainerDispatchError(RuntimeError):
    """The EAAEF container boundary rejected a dispatch or receipt."""


class ExternalAgentContainerDispatchUnavailable(ExternalAgentContainerDispatchError):
    """Required owner, profile, provider, or verifier authority is absent."""


class ExternalAgentContainerDispatchAmbiguous(ExternalAgentContainerDispatchError):
    """A reserved effect may have run and therefore cannot be replayed."""


class ExternalAgentContainerMergeAdmissionPending(ExternalAgentContainerDispatchError):
    """A verified patch is waiting for independent host merge admission."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_cid(value: Any, name: str) -> str:
    text = str(value or "")
    if _CID.fullmatch(text) is None:
        raise ExternalAgentContainerDispatchError(f"{name} must be a sha256 CID")
    return text


def _require_id(value: Any, name: str) -> str:
    text = str(value or "")
    if _BOUNDED_ID.fullmatch(text) is None:
        raise ExternalAgentContainerDispatchError(f"{name} is not a bounded identity")
    return text


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ExternalAgentContainerDispatchError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ExternalAgentContainerDispatchError(
            f"{name} must be a positive integer"
        ) from exc
    if result < 1:
        raise ExternalAgentContainerDispatchError(f"{name} must be a positive integer")
    return result


def _require_cid_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        raise ExternalAgentContainerDispatchError(f"{name} must be a CID list")
    result = tuple(_require_cid(item, name) for item in value)
    if len(result) != len(set(result)) or tuple(sorted(result)) != result:
        raise ExternalAgentContainerDispatchError(
            f"{name} must be unique and canonically sorted"
        )
    return result


@dataclass(frozen=True)
class ExternalAgentContainerWorkPacket:
    """Path-free identity for one exact container mutation attempt."""

    task_id: str
    task_cid: str
    attempt_id: str
    attempt_number: int
    plan_revision_cid: str
    repository_tree: str
    semantic_state_root: str
    worktree_id: str
    planned_container_id: str
    worker_principal_did: str
    provider_principal_did: str
    provider: str
    model_route_cid: str
    container_profile_cid: str
    image_digest: str
    network_authorization_cid: str
    lease_id: str
    fencing_token: int
    fence_epoch: int
    idempotency_key: str
    effect_scope_cid: str
    gateway_binding_cid: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExternalAgentContainerWorkPacket:
        """Decode a closed, content-addressed packet without host paths."""

        required = {
            "schema",
            "interface",
            "board_namespace",
            "task_id",
            "task_cid",
            "attempt_id",
            "attempt_number",
            "plan_revision_cid",
            "repository_tree",
            "semantic_state_root",
            "worktree_id",
            "planned_container_id",
            "worker_principal_did",
            "provider_principal_did",
            "provider",
            "model_route_cid",
            "container_profile_cid",
            "image_digest",
            "network_authorization_cid",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "idempotency_key",
            "effect_scope_cid",
            "gateway_binding_cid",
            "packet_cid",
        }
        payload = dict(value)
        if set(payload) != required:
            raise ExternalAgentContainerDispatchError(
                "container work packet shape is not exact"
            )
        body = {key: item for key, item in payload.items() if key != "packet_cid"}
        if (
            payload.get("schema") != EXTERNAL_AGENT_CONTAINER_WORK_PACKET_SCHEMA
            or payload.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or payload.get("board_namespace") != EAAEF_BOARD_NAMESPACE
            or payload.get("packet_cid") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchError(
                "container work packet identity is invalid"
            )
        repository_tree = str(payload.get("repository_tree") or "")
        if _GIT_TREE.fullmatch(repository_tree) is None:
            raise ExternalAgentContainerDispatchError(
                "repository_tree must be an exact Git tree or sha256 CID"
            )
        provider = str(payload.get("provider") or "").lower()
        worker_did = str(payload.get("worker_principal_did") or "")
        provider_did = str(payload.get("provider_principal_did") or "")
        if (
            provider not in _PROVIDERS
            or not worker_did.startswith("did:key:z")
            or not provider_did.startswith("did:key:z")
            or worker_did == provider_did
        ):
            raise ExternalAgentContainerDispatchError(
                "container work packet principal/provider binding is invalid"
            )
        return cls(
            task_id=_require_id(payload["task_id"], "task_id"),
            task_cid=_require_cid(payload["task_cid"], "task_cid"),
            attempt_id=_require_id(payload["attempt_id"], "attempt_id"),
            attempt_number=_require_positive_int(
                payload["attempt_number"], "attempt_number"
            ),
            plan_revision_cid=_require_cid(
                payload["plan_revision_cid"], "plan_revision_cid"
            ),
            repository_tree=repository_tree,
            semantic_state_root=_require_cid(
                payload["semantic_state_root"], "semantic_state_root"
            ),
            worktree_id=_require_cid(payload["worktree_id"], "worktree_id"),
            planned_container_id=_require_cid(
                payload["planned_container_id"], "planned_container_id"
            ),
            worker_principal_did=worker_did,
            provider_principal_did=provider_did,
            provider=provider,
            model_route_cid=_require_cid(
                payload["model_route_cid"], "model_route_cid"
            ),
            container_profile_cid=_require_cid(
                payload["container_profile_cid"], "container_profile_cid"
            ),
            image_digest=_require_cid(payload["image_digest"], "image_digest"),
            network_authorization_cid=_require_cid(
                payload["network_authorization_cid"],
                "network_authorization_cid",
            ),
            lease_id=_require_id(payload["lease_id"], "lease_id"),
            fencing_token=_require_positive_int(
                payload["fencing_token"], "fencing_token"
            ),
            fence_epoch=_require_positive_int(payload["fence_epoch"], "fence_epoch"),
            idempotency_key=_require_id(
                payload["idempotency_key"], "idempotency_key"
            ),
            effect_scope_cid=_require_cid(
                payload["effect_scope_cid"], "effect_scope_cid"
            ),
            gateway_binding_cid=_require_cid(
                payload["gateway_binding_cid"], "gateway_binding_cid"
            ),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_AGENT_CONTAINER_WORK_PACKET_SCHEMA,
            "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "board_namespace": EAAEF_BOARD_NAMESPACE,
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "plan_revision_cid": self.plan_revision_cid,
            "repository_tree": self.repository_tree,
            "semantic_state_root": self.semantic_state_root,
            "worktree_id": self.worktree_id,
            "planned_container_id": self.planned_container_id,
            "worker_principal_did": self.worker_principal_did,
            "provider_principal_did": self.provider_principal_did,
            "provider": self.provider,
            "model_route_cid": self.model_route_cid,
            "container_profile_cid": self.container_profile_cid,
            "image_digest": self.image_digest,
            "network_authorization_cid": self.network_authorization_cid,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "fence_epoch": self.fence_epoch,
            "idempotency_key": self.idempotency_key,
            "effect_scope_cid": self.effect_scope_cid,
            "gateway_binding_cid": self.gateway_binding_cid,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.body()
        return {**body, "packet_cid": _content_id(body)}

    @property
    def packet_cid(self) -> str:
        return _content_id(self.body())


PacketProvider = Callable[[Any], ExternalAgentContainerWorkPacket]
QualificationGuard = Callable[[ExternalAgentContainerWorkPacket], Mapping[str, Any]]
ContainerLauncher = Callable[
    [ExternalAgentContainerWorkPacket, Mapping[str, Any]], Mapping[str, Any]
]
IndependentVerifier = Callable[
    [ExternalAgentContainerWorkPacket, Mapping[str, Any]], Mapping[str, Any]
]
MergeAdmissionObserver = Callable[
    [ExternalAgentContainerWorkPacket, Mapping[str, Any]], Mapping[str, Any] | None
]
HostSourceObserver = Callable[[], str]


class ExternalAgentContainerWorkerDispatcher:
    """Reserve, dispatch, verify, and expose a merge-inert worker proposal."""

    INTERFACE = EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE

    def __init__(
        self,
        *,
        execution_repository: Any,
        packet_provider: PacketProvider,
        qualification_guard: QualificationGuard,
        container_launcher: ContainerLauncher,
        independent_verifier: IndependentVerifier,
        merge_admission_observer: MergeAdmissionObserver,
        host_source_observer: HostSourceObserver,
        now_ms: Callable[[], int],
    ) -> None:
        required_methods = (
            "reserve_effect",
            "commit_effect",
        )
        if any(
            not callable(getattr(execution_repository, method, None))
            for method in required_methods
        ):
            raise ExternalAgentContainerDispatchUnavailable(
                "Quack execution owner lacks the closed reserve/commit surface"
            )
        callbacks = (
            packet_provider,
            qualification_guard,
            container_launcher,
            independent_verifier,
            merge_admission_observer,
            host_source_observer,
            now_ms,
        )
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("container dispatcher callbacks must all be callable")
        self._execution_repository = execution_repository
        self._packet_provider = packet_provider
        self._qualification_guard = qualification_guard
        self._container_launcher = container_launcher
        self._independent_verifier = independent_verifier
        self._merge_admission_observer = merge_admission_observer
        self._host_source_observer = host_source_observer
        self._now_ms = now_ms

    @staticmethod
    def _match_attempt(packet: ExternalAgentContainerWorkPacket, attempt: Any) -> None:
        checks = {
            "task_cid": str(getattr(attempt, "task_cid", "") or ""),
            "attempt_id": str(getattr(attempt, "attempt_id", "") or ""),
            "attempt_number": int(getattr(attempt, "attempt_number", 0) or 0),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "fencing_token": int(getattr(attempt, "fencing_token", 0) or 0),
            "fence_epoch": int(getattr(attempt, "fence_epoch", 0) or 0),
        }
        mismatched = [
            name for name, observed in checks.items() if getattr(packet, name) != observed
        ]
        if mismatched:
            raise ExternalAgentContainerDispatchError(
                "work packet differs from the fenced attempt: " + ", ".join(mismatched)
            )

    @staticmethod
    def _dispatch_claim(packet: ExternalAgentContainerWorkPacket) -> dict[str, Any]:
        body = {
            "schema": EXTERNAL_AGENT_CONTAINER_DISPATCH_CLAIM_SCHEMA,
            "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "packet_cid": packet.packet_cid,
            "task_id": packet.task_id,
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "attempt_number": packet.attempt_number,
            "plan_revision_cid": packet.plan_revision_cid,
            "repository_tree": packet.repository_tree,
            "semantic_state_root": packet.semantic_state_root,
            "worktree_id": packet.worktree_id,
            "planned_container_id": packet.planned_container_id,
            "container_profile_cid": packet.container_profile_cid,
            "image_digest": packet.image_digest,
            "network_authorization_cid": packet.network_authorization_cid,
            "lease_id": packet.lease_id,
            "fencing_token": packet.fencing_token,
            "fence_epoch": packet.fence_epoch,
            "idempotency_key": packet.idempotency_key,
            "effect_scope_cid": packet.effect_scope_cid,
            "gateway_binding_cid": packet.gateway_binding_cid,
            "worker_principal_did": packet.worker_principal_did,
            "provider_principal_did": packet.provider_principal_did,
            "provider": packet.provider,
            "model_route_cid": packet.model_route_cid,
        }
        return {**body, "claim_cid": _content_id(body)}

    @staticmethod
    def _verify_qualification(
        packet: ExternalAgentContainerWorkPacket,
        value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        required = {
            "status",
            "dispatcher_interface",
            "gateway_binding_cid",
            "container_profile_cid",
            "image_digest",
            "reservation_adapter_status",
            "container_launcher_status",
            "independent_verifier_status",
            "host_source_isolation_status",
            "qualification_receipt_cid",
        }
        result = dict(value)
        if (
            set(result) != required
            or result.get("status") != "admitted"
            or result.get("dispatcher_interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or result.get("gateway_binding_cid") != packet.gateway_binding_cid
            or result.get("container_profile_cid") != packet.container_profile_cid
            or result.get("image_digest") != packet.image_digest
            or any(
                result.get(name) != "qualified"
                for name in (
                    "reservation_adapter_status",
                    "container_launcher_status",
                    "independent_verifier_status",
                    "host_source_isolation_status",
                )
            )
            or _CID.fullmatch(str(result.get("qualification_receipt_cid") or ""))
            is None
        ):
            raise ExternalAgentContainerDispatchUnavailable(
                "container dispatcher qualification is absent or divergent"
            )
        return MappingProxyType(result)

    @staticmethod
    def _verify_reservation(
        value: Mapping[str, Any],
        *,
        claim_cid: str,
    ) -> Mapping[str, Any]:
        required = {
            "schema",
            "claim_cid",
            "reservation_id",
            "outcome",
            "reason_codes",
            "accepted_result",
            "receipt_cid",
        }
        receipt = dict(value)
        body = {key: item for key, item in receipt.items() if key != "receipt_cid"}
        if (
            set(receipt) != required
            or receipt.get("schema")
            != EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA
            or receipt.get("claim_cid") != claim_cid
            or _CID.fullmatch(str(receipt.get("reservation_id") or "")) is None
            or not isinstance(receipt.get("reason_codes"), list)
            or any(not isinstance(item, str) for item in receipt["reason_codes"])
            or receipt.get("receipt_cid") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchUnavailable(
                "Quack owner returned an unqualified dispatch reservation"
            )
        outcome = str(receipt.get("outcome") or "")
        if outcome not in {
            "reserved_new",
            "accepted_replay",
            "in_flight_ambiguous",
            "denied",
            "unavailable_fail_closed",
        }:
            raise ExternalAgentContainerDispatchUnavailable(
                "Quack owner returned an unknown reservation outcome"
            )
        if outcome == "reserved_new" and receipt.get("accepted_result") is not None:
            raise ExternalAgentContainerDispatchUnavailable(
                "new reservation unexpectedly contains an accepted result"
            )
        if outcome == "accepted_replay" and not isinstance(
            receipt.get("accepted_result"), Mapping
        ):
            raise ExternalAgentContainerDispatchUnavailable(
                "accepted replay lacks its exact result"
            )
        return MappingProxyType(receipt)

    @staticmethod
    def _verify_proposal(
        packet: ExternalAgentContainerWorkPacket,
        claim: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        required = {
            "schema",
            "interface",
            "status",
            "claim_cid",
            "packet_cid",
            "task_cid",
            "attempt_id",
            "worker_principal_did",
            "provider_principal_did",
            "image_digest",
            "container_profile_cid",
            "network_authorization_cid",
            "runtime_container_id",
            "patch_artifact_cid",
            "artifact_cids",
            "test_receipt_cids",
            "proof_receipt_cids",
            "host_source_mutated",
            "host_merge_attempted",
            "push_attempted",
            "receipt_cid",
        }
        receipt = dict(value)
        body = {key: item for key, item in receipt.items() if key != "receipt_cid"}
        if (
            set(receipt) != required
            or receipt.get("schema") != EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA
            or receipt.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or receipt.get("status") != "proposal_ready"
            or receipt.get("claim_cid") != claim["claim_cid"]
            or receipt.get("packet_cid") != packet.packet_cid
            or receipt.get("task_cid") != packet.task_cid
            or receipt.get("attempt_id") != packet.attempt_id
            or receipt.get("worker_principal_did") != packet.worker_principal_did
            or receipt.get("provider_principal_did") != packet.provider_principal_did
            or receipt.get("image_digest") != packet.image_digest
            or receipt.get("container_profile_cid") != packet.container_profile_cid
            or receipt.get("network_authorization_cid")
            != packet.network_authorization_cid
            or receipt.get("host_source_mutated") is not False
            or receipt.get("host_merge_attempted") is not False
            or receipt.get("push_attempted") is not False
            or receipt.get("receipt_cid") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchError(
                "container proposal receipt is unbound or claims a host effect"
            )
        for name in (
            "runtime_container_id",
            "patch_artifact_cid",
        ):
            _require_cid(receipt.get(name), name)
        for name in ("artifact_cids", "test_receipt_cids", "proof_receipt_cids"):
            _require_cid_list(receipt.get(name), name)
        return MappingProxyType(receipt)

    @staticmethod
    def _verify_independent_receipt(
        packet: ExternalAgentContainerWorkPacket,
        proposal: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        required = {
            "schema",
            "interface",
            "outcome",
            "claim_cid",
            "proposal_receipt_cid",
            "verifier_principal_did",
            "test_receipt_cids",
            "proof_receipt_cids",
            "receipt_cid",
        }
        receipt = dict(value)
        body = {key: item for key, item in receipt.items() if key != "receipt_cid"}
        verifier = str(receipt.get("verifier_principal_did") or "")
        if (
            set(receipt) != required
            or receipt.get("schema")
            != EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA
            or receipt.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or receipt.get("outcome") != "passed"
            or receipt.get("claim_cid") != proposal["claim_cid"]
            or receipt.get("proposal_receipt_cid") != proposal["receipt_cid"]
            or not verifier.startswith("did:key:z")
            or verifier
            in {packet.worker_principal_did, packet.provider_principal_did}
            or receipt.get("receipt_cid") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchError(
                "worker proposal lacks independent passing verification"
            )
        verified_tests = _require_cid_list(
            receipt.get("test_receipt_cids"), "test_receipt_cids"
        )
        verified_proofs = _require_cid_list(
            receipt.get("proof_receipt_cids"), "proof_receipt_cids"
        )
        if (
            verified_tests != tuple(proposal["test_receipt_cids"])
            or verified_proofs != tuple(proposal["proof_receipt_cids"])
        ):
            raise ExternalAgentContainerDispatchError(
                "independent verification did not cover exact worker receipts"
            )
        return MappingProxyType(receipt)

    @staticmethod
    def _accepted_result(
        packet: ExternalAgentContainerWorkPacket,
        claim: Mapping[str, Any],
        reservation: Mapping[str, Any],
        proposal: Mapping[str, Any],
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = {
            "schema": EXTERNAL_AGENT_CONTAINER_ACCEPTED_RESULT_SCHEMA,
            "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "status": "succeeded",
            # Compatibility with DatabaseImplementationDaemon means only that
            # the provider invocation is admitted for the next phase.  It is
            # explicitly not task-completion or merge acceptance.
            "accepted": True,
            "task_result_accepted": False,
            "merge_admitted": False,
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "packet_cid": packet.packet_cid,
            "claim_cid": claim["claim_cid"],
            "reservation_id": reservation["reservation_id"],
            "proposal_receipt_cid": proposal["receipt_cid"],
            "verification_receipt_cid": verification["receipt_cid"],
            "patch_artifact_cid": proposal["patch_artifact_cid"],
            "artifact_cids": list(proposal["artifact_cids"]),
            "test_receipt_cids": list(proposal["test_receipt_cids"]),
            "proof_receipt_cids": list(proposal["proof_receipt_cids"]),
            "worker_principal_did": packet.worker_principal_did,
            "independent_verifier_principal_did": verification[
                "verifier_principal_did"
            ],
        }
        return {**body, "receipt_id": _content_id(body)}

    @classmethod
    def _verify_accepted_result(
        cls,
        packet: ExternalAgentContainerWorkPacket,
        claim: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        required = {
            "schema",
            "interface",
            "status",
            "accepted",
            "task_result_accepted",
            "merge_admitted",
            "task_cid",
            "attempt_id",
            "packet_cid",
            "claim_cid",
            "reservation_id",
            "proposal_receipt_cid",
            "verification_receipt_cid",
            "patch_artifact_cid",
            "artifact_cids",
            "test_receipt_cids",
            "proof_receipt_cids",
            "worker_principal_did",
            "independent_verifier_principal_did",
            "receipt_id",
        }
        result = dict(value)
        body = {key: item for key, item in result.items() if key != "receipt_id"}
        verifier = str(result.get("independent_verifier_principal_did") or "")
        if (
            set(result) != required
            or result.get("schema") != EXTERNAL_AGENT_CONTAINER_ACCEPTED_RESULT_SCHEMA
            or result.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or result.get("status") != "succeeded"
            or result.get("accepted") is not True
            or result.get("task_result_accepted") is not False
            or result.get("merge_admitted") is not False
            or result.get("task_cid") != packet.task_cid
            or result.get("attempt_id") != packet.attempt_id
            or result.get("packet_cid") != packet.packet_cid
            or result.get("claim_cid") != claim["claim_cid"]
            or result.get("worker_principal_did") != packet.worker_principal_did
            or verifier in {packet.worker_principal_did, packet.provider_principal_did}
            or not verifier.startswith("did:key:z")
            or result.get("receipt_id") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchError(
                "accepted container result is divergent or self-approved"
            )
        for name in (
            "reservation_id",
            "proposal_receipt_cid",
            "verification_receipt_cid",
            "patch_artifact_cid",
        ):
            _require_cid(result.get(name), name)
        for name in ("artifact_cids", "test_receipt_cids", "proof_receipt_cids"):
            _require_cid_list(result.get(name), name)
        return MappingProxyType(result)

    def _packet(self, attempt: Any) -> ExternalAgentContainerWorkPacket:
        packet = self._packet_provider(attempt)
        if not isinstance(packet, ExternalAgentContainerWorkPacket):
            raise ExternalAgentContainerDispatchError(
                "packet provider did not return a typed work packet"
            )
        # Round-trip the closed schema on every effect boundary.
        packet = ExternalAgentContainerWorkPacket.from_mapping(packet.to_dict())
        self._match_attempt(packet, attempt)
        self._verify_qualification(packet, self._qualification_guard(packet))
        return packet

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Reserve before launch and return an independently verified proposal."""

        packet = self._packet(attempt)
        claim = self._dispatch_claim(packet)
        reservation_raw = self._execution_repository.reserve_effect(
            kind="external_agent_container_dispatch",
            record_id=claim["claim_cid"],
            attempt_id=packet.attempt_id,
            task_cid=packet.task_cid,
            operation_key=claim["claim_cid"],
            idempotency_key=packet.idempotency_key,
            owner_session_id=str(getattr(attempt, "owner_session_id", "") or ""),
            recorded_at_ms=int(self._now_ms()),
            fencing_token=packet.fencing_token,
            fence_epoch=packet.fence_epoch,
            claim=claim,
        )
        if not isinstance(reservation_raw, Mapping):
            raise ExternalAgentContainerDispatchUnavailable(
                "Quack owner did not return a typed pre-effect reservation"
            )
        reservation = self._verify_reservation(
            reservation_raw, claim_cid=claim["claim_cid"]
        )
        outcome = str(reservation["outcome"])
        if outcome == "accepted_replay":
            return self._verify_accepted_result(
                packet,
                claim,
                reservation["accepted_result"],
            )
        if outcome == "in_flight_ambiguous":
            raise ExternalAgentContainerDispatchAmbiguous(
                "container dispatch is durably reserved but has no accepted receipt; "
                "automatic replay is forbidden"
            )
        if outcome != "reserved_new":
            reasons = ",".join(str(item) for item in reservation["reason_codes"])
            raise ExternalAgentContainerDispatchUnavailable(
                "container dispatch reservation was not admitted"
                + (f": {reasons}" if reasons else "")
            )

        before = str(self._host_source_observer() or "")
        if not before:
            raise ExternalAgentContainerDispatchUnavailable(
                "host source immutability observer is unavailable"
            )
        try:
            proposal_raw = self._container_launcher(packet, reservation)
        except Exception as exc:
            after_failure = str(self._host_source_observer() or "")
            if after_failure != before:
                raise ExternalAgentContainerDispatchError(
                    "host source changed during a failed container dispatch"
                ) from exc
            raise ExternalAgentContainerDispatchAmbiguous(
                "container launch failed after durable reservation; automatic "
                "retry requires owner-side adoption or compensation"
            ) from exc
        after = str(self._host_source_observer() or "")
        if after != before:
            raise ExternalAgentContainerDispatchError(
                "container dispatcher observed forbidden host source mutation"
            )
        if not isinstance(proposal_raw, Mapping):
            raise ExternalAgentContainerDispatchError(
                "container launcher returned a non-object proposal"
            )
        proposal = self._verify_proposal(packet, claim, proposal_raw)
        verification_raw = self._independent_verifier(packet, proposal)
        if not isinstance(verification_raw, Mapping):
            raise ExternalAgentContainerDispatchError(
                "independent verifier returned a non-object receipt"
            )
        verification = self._verify_independent_receipt(
            packet, proposal, verification_raw
        )
        accepted = self._accepted_result(
            packet,
            claim,
            reservation,
            proposal,
            verification,
        )
        committed = self._execution_repository.commit_effect(
            kind="external_agent_container_dispatch",
            record_id=claim["claim_cid"],
            attempt_id=packet.attempt_id,
            task_cid=packet.task_cid,
            operation_key=claim["claim_cid"],
            idempotency_key=packet.idempotency_key,
            owner_session_id=str(getattr(attempt, "owner_session_id", "") or ""),
            recorded_at_ms=int(self._now_ms()),
            fencing_token=packet.fencing_token,
            fence_epoch=packet.fence_epoch,
            claim=claim,
            reservation_id=reservation["reservation_id"],
            result=accepted,
        )
        if not isinstance(committed, Mapping):
            raise ExternalAgentContainerDispatchAmbiguous(
                "container result commit has an ambiguous owner response"
            )
        return self._verify_accepted_result(packet, claim, committed)

    def apply_effect(
        self,
        attempt: Any,
        provider_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Record a merge-inert patch proposal; never mutate host source."""

        packet = self._packet(attempt)
        claim = self._dispatch_claim(packet)
        accepted = self._verify_accepted_result(packet, claim, provider_result)
        body = {
            "schema": EXTERNAL_AGENT_CONTAINER_EFFECT_RECEIPT_SCHEMA,
            "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "status": "applied",
            "effect": "isolated_container_patch_proposal_recorded",
            "effect_key": "eaaef-proposal:" + claim["claim_cid"],
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "packet_cid": packet.packet_cid,
            "claim_cid": claim["claim_cid"],
            "accepted_result_receipt_id": accepted["receipt_id"],
            "patch_artifact_cid": accepted["patch_artifact_cid"],
            "task_result_accepted": False,
            "merge_admitted": False,
            "host_mutation_performed": False,
        }
        return {**body, "receipt_cid": _content_id(body)}

    def validate_effect(
        self,
        attempt: Any,
        effect_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Require a separate, independent host merge/deliverable admission."""

        packet = self._packet(attempt)
        claim = self._dispatch_claim(packet)
        effect = dict(effect_result)
        body = {key: item for key, item in effect.items() if key != "receipt_cid"}
        if (
            effect.get("schema") != EXTERNAL_AGENT_CONTAINER_EFFECT_RECEIPT_SCHEMA
            or effect.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or effect.get("status") != "applied"
            or effect.get("claim_cid") != claim["claim_cid"]
            or effect.get("task_cid") != packet.task_cid
            or effect.get("attempt_id") != packet.attempt_id
            or effect.get("task_result_accepted") is not False
            or effect.get("merge_admitted") is not False
            or effect.get("host_mutation_performed") is not False
            or effect.get("receipt_cid") != _content_id(body)
        ):
            raise ExternalAgentContainerDispatchError(
                "container effect receipt is divergent or claims host authority"
            )
        admission_raw = self._merge_admission_observer(packet, effect)
        if admission_raw is None:
            raise ExternalAgentContainerMergeAdmissionPending(
                "verified patch has no independent host merge/deliverable admission"
            )
        admission = dict(admission_raw)
        required = {
            "schema",
            "interface",
            "decision",
            "delivery_mode",
            "task_cid",
            "attempt_id",
            "claim_cid",
            "accepted_result_receipt_id",
            "patch_artifact_cid",
            "reviewer_principal_did",
            "effect_authority_cid",
            "merge_commit",
            "receipt_cid",
        }
        admission_body = {
            key: item for key, item in admission.items() if key != "receipt_cid"
        }
        reviewer = str(admission.get("reviewer_principal_did") or "")
        delivery_mode = str(admission.get("delivery_mode") or "")
        merge_commit = str(admission.get("merge_commit") or "")
        if (
            set(admission) != required
            or admission.get("schema") != EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA
            or admission.get("interface")
            != EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE
            or admission.get("decision") != "accepted"
            or delivery_mode not in {"merge_accepted", "reviewed_patch"}
            or admission.get("task_cid") != packet.task_cid
            or admission.get("attempt_id") != packet.attempt_id
            or admission.get("claim_cid") != claim["claim_cid"]
            or admission.get("accepted_result_receipt_id")
            != effect.get("accepted_result_receipt_id")
            or admission.get("patch_artifact_cid")
            != effect.get("patch_artifact_cid")
            or not reviewer.startswith("did:key:z")
            or reviewer in {packet.worker_principal_did, packet.provider_principal_did}
            or (delivery_mode == "merge_accepted" and re.fullmatch(r"[0-9a-f]{40}", merge_commit) is None)
            or (delivery_mode == "reviewed_patch" and merge_commit)
            or admission.get("receipt_cid") != _content_id(admission_body)
        ):
            raise ExternalAgentContainerDispatchError(
                "host merge admission is divergent or self-approved"
            )
        authority_cid = _require_cid(
            admission.get("effect_authority_cid"), "effect_authority_cid"
        )
        control_claim_id = _require_id(
            getattr(attempt, "claim_id", ""), "control claim_id"
        )
        owner_session_id = _require_id(
            getattr(attempt, "owner_session_id", ""), "owner_session_id"
        )
        return MappingProxyType(
            {
                "outcome": "passed",
                "evidence_digest": admission["receipt_cid"],
                "argv": ["external-agent-host-merge-admission"],
                "body": {
                    "schema": EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA,
                    "validator": self.INTERFACE,
                    "task_cid": packet.task_cid,
                    "attempt_id": packet.attempt_id,
                    "control_claim_id": control_claim_id,
                    "dispatch_claim_cid": claim["claim_cid"],
                    "owner_session_id": owner_session_id,
                    "fencing_token": packet.fencing_token,
                    "fence_epoch": packet.fence_epoch,
                    "authority_cid": authority_cid,
                    "admission_receipt": admission,
                    "delivery_mode": delivery_mode,
                    "merge_commit": merge_commit,
                    "patch_artifact_cid": admission["patch_artifact_cid"],
                },
            }
        )

    @staticmethod
    def qualification_evidence() -> Mapping[str, Any]:
        """Return current source truth without promoting unavailable adapters."""

        return MappingProxyType(
            {
                "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "status": EXTERNAL_AGENT_CONTAINER_DISPATCH_STATUS,
                "blockers": list(EXTERNAL_AGENT_CONTAINER_DISPATCH_BLOCKERS),
                "portal_fallback_allowed": False,
                "host_provider_allowed": False,
                "host_merge_inside_worker_allowed": False,
            }
        )


__all__ = (
    "EXTERNAL_AGENT_CONTAINER_ACCEPTED_RESULT_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_DISPATCH_BLOCKERS",
    "EXTERNAL_AGENT_CONTAINER_DISPATCH_CLAIM_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_DISPATCH_STATUS",
    "EXTERNAL_AGENT_CONTAINER_EFFECT_RECEIPT_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_WORK_PACKET_SCHEMA",
    "EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE",
    "EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA",
    "ExternalAgentContainerDispatchAmbiguous",
    "ExternalAgentContainerDispatchError",
    "ExternalAgentContainerDispatchUnavailable",
    "ExternalAgentContainerMergeAdmissionPending",
    "ExternalAgentContainerWorkPacket",
    "ExternalAgentContainerWorkerDispatcher",
)
