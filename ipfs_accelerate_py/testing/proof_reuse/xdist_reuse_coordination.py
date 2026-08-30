"""PCTDD-030: extend xdist controller/worker proof-reuse coordination.

Workers return bounded public intents.  Only the controller may publish
accepted reuse evidence through the existing fenced xdist coordinator.
V2 execution-key and optional composite-phase-receipt pins travel as public
identity only.  They never authorize pytest skip, never carry witness
material, and never let a worker write stores.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ...agent_supervisor.proof.test_execution_contracts import TestPassReceipt
from .xdist import (
    COORDINATION_UNAVAILABLE,
    MAX_PACKET_BYTES,
    PROOF_REUSE_XDIST_INTERFACE,
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
    ProofReuseXdistRole,
    _bounded_token,
    _public_deferred_request,
)

# This module is accelerate integration, not a pytest test module.
__test__ = False

XDIST_REUSE_COORDINATION_INTERFACE: Final = "XdistReuseCoordination@1"
XDIST_REUSE_COORDINATION_RESULT_INTERFACE: Final = (
    "XdistReuseCoordinationResult@1"
)
XDIST_REUSE_INTENT_INTERFACE: Final = "XdistReuseIntent@1"
XDIST_REUSE_COORDINATION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/xdist-reuse-coordination@1"
)
XDIST_REUSE_COORDINATION_POLICY_INTERFACE: Final = (
    "XdistReuseCoordinationPolicy@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
COORDINATION_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.xdist_reuse_coordination"
)
PREDECESSOR_INTERFACE: Final = PROOF_REUSE_XDIST_INTERFACE
ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_composite_phase_receipt"
)
ITEM_COMPOSITE_PHASE_RECEIPT_CID_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_composite_phase_receipt_cid"
)
MAX_BOUNDED_INTENT_BYTES: Final = MAX_PACKET_BYTES
MAX_TEXT_CHARS: Final = 4_096
_DIGEST_PREFIX: Final = "sha256:"

COORDINATION_ESTABLISHES: Final = (
    "workers return bounded intents and only the controller may publish "
    "accepted reuse evidence"
)
COORDINATION_DOES_NOT: Final = (
    "execution or semantics; skip; worker publication; current-root "
    "publication; production ZK; guarded post-setup reuse; pre-setup item "
    "reuse; signed admission; fixture-proof-aware scheduling"
)

_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private",
    "proving_key",
    "secret",
    "session",
    "signing_key",
    "token",
    "witness",
)

_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "guarded_post_setup_reuse",
        "guarded_post_setup_reuse_not_implemented",
        "xdist reuse coordination transports bounded public intents only; "
        "guarded post-setup pre-call reuse remains a later accelerate task",
    ),
    (
        "pre_setup_item_reuse",
        "pre_setup_item_reuse_not_implemented",
        "xdist reuse coordination does not gate whole-item pre-setup reuse; "
        "that remains a later accelerate task",
    ),
    (
        "fixture_proof_aware_xdist",
        "fixture_proof_aware_xdist_not_implemented",
        "fixture affinity and proof-cost placement remain a later xdist "
        "scheduling task; coordination does not omit tests or merge pools",
    ),
    (
        "signed_runner_attestation",
        "signed_runner_attestation_not_verified",
        "bounded intents pin public identity only; runner signature "
        "verification remains a later accelerate attestation task",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; xdist coordination "
        "cannot admit simulated, structural, or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by xdist reuse "
        "coordination",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade xdist reuse-coordination integrity commitments",
    ),
)


class XdistReuseCoordinationError(ValueError):
    """Raised when xdist reuse coordination is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise XdistReuseCoordinationError(
            "floating-point values are not JSON-safe for xdist reuse intents"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise XdistReuseCoordinationError(
                    f"xdist reuse intent rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise XdistReuseCoordinationError(
            "xdist reuse intent rejects secret or raw bytes"
        )
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise XdistReuseCoordinationError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def canonical_public_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a privacy-safe public payload."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def public_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` of canonical public bytes."""

    return _DIGEST_PREFIX + hashlib.sha256(canonical_public_bytes(value)).hexdigest()


COORDINATION_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": XDIST_REUSE_COORDINATION_POLICY_INTERFACE,
        "coordination_interface": XDIST_REUSE_COORDINATION_INTERFACE,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "workers_may_publish": False,
        "controller_owns_writes": True,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "normal_execution_fallback": True,
        "schema_authority": COORDINATION_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(COORDINATION_POLICY))


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise XdistReuseCoordinationError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-030 typed unavailable capabilities."""

    return tuple(
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    )


def authority_descriptor() -> dict[str, Any]:
    """Return the coordination authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "coordination": COORDINATION_AUTHORITY,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "does_not": COORDINATION_DOES_NOT,
        "establishes": COORDINATION_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "workers_may_publish": False,
        "controller_owns_writes": True,
        "normal_execution_fallback": True,
        "coordination_interface": XDIST_REUSE_COORDINATION_INTERFACE,
        "intent_interface": XDIST_REUSE_INTENT_INTERFACE,
        "test_execution_key_v2": "TestExecutionKeyV2",
        "composite_phase_receipt": "CompositePhaseReceipt@1",
    }


def workers_may_publish() -> bool:
    """Workers never publish accepted reuse evidence."""

    return False


def controller_owns_publication(role: ProofReuseXdistRole | str) -> bool:
    """Return whether *role* is the sole write authority."""

    typed = role if isinstance(role, ProofReuseXdistRole) else ProofReuseXdistRole(role)
    return typed in (
        ProofReuseXdistRole.CONTROLLER,
        ProofReuseXdistRole.STANDALONE,
    )


def composite_phase_receipt_cid_from_item(item: Any) -> str:
    """Return a public composite-phase-receipt CID pin from *item*, if any."""

    if item is None:
        return ""
    raw = getattr(item, ITEM_COMPOSITE_PHASE_RECEIPT_CID_ATTRIBUTE, "") or ""
    pinned = _bounded_token(raw)
    if raw and not pinned:
        raise XdistReuseCoordinationError(
            "composite phase receipt cid is not a public pin"
        )
    if pinned:
        return pinned
    attached = getattr(item, ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE, None)
    if attached is None:
        return ""
    for attr in ("receipt_cid", "composite_phase_receipt_cid"):
        value = getattr(attached, attr, None)
        if isinstance(value, str) and value:
            pinned = _bounded_token(value)
            if not pinned:
                raise XdistReuseCoordinationError(
                    "composite phase receipt cid is not a public pin"
                )
            return pinned
    if isinstance(attached, Mapping):
        for key in ("receipt_cid", "composite_phase_receipt_cid"):
            value = attached.get(key)
            if isinstance(value, str) and value:
                pinned = _bounded_token(value)
                if not pinned:
                    raise XdistReuseCoordinationError(
                        "composite phase receipt cid is not a public pin"
                    )
                return pinned
    return ""


def _attached_execution_key_cid(item: Any) -> str:
    if item is None:
        return ""
    try:
        from .setup_bound_execution_key import get_attached_setup_bound_execution_key

        key = get_attached_setup_bound_execution_key(item)
    except Exception:
        key = getattr(item, "_ipfs_proof_reuse_setup_bound_execution_key_v2", None)
    if key is None:
        return ""
    for attr in ("execution_key_cid", "content_id"):
        value = getattr(key, attr, None)
        if isinstance(value, str) and value:
            pinned = _bounded_token(value)
            if not pinned:
                raise XdistReuseCoordinationError(
                    "attached execution key cid is not a public pin"
                )
            return pinned
    if isinstance(key, Mapping):
        value = key.get("execution_key_cid") or key.get("content_id") or ""
        if isinstance(value, str) and value:
            pinned = _bounded_token(value)
            if not pinned:
                raise XdistReuseCoordinationError(
                    "attached execution key cid is not a public pin"
                )
            return pinned
    return ""


def _bound_intent_size(payload: Mapping[str, Any]) -> None:
    encoded = canonical_public_bytes(payload)
    if len(encoded) > MAX_BOUNDED_INTENT_BYTES:
        raise XdistReuseCoordinationError("xdist reuse intent is over budget")
    if len(encoded) > MAX_PACKET_BYTES:
        raise XdistReuseCoordinationError("xdist reuse intent is over packet budget")


def bound_worker_intent(
    receipt: TestPassReceipt | Mapping[str, Any] | ProofReusePublicationIntent,
    *,
    certificate: Mapping[str, Any] | None = None,
    certificate_cid: str = "",
    deferred_request: Mapping[str, Any] | None = None,
    execution_key_cid: str = "",
    composite_phase_receipt_cid: str = "",
    item: Any = None,
) -> ProofReusePublicationIntent:
    """Return one public-only worker intent or raise."""

    if deferred_request is not None:
        _json_ready(deferred_request)
    if certificate is not None:
        _json_ready(certificate)
    if isinstance(receipt, ProofReusePublicationIntent):
        intent = receipt.validated()
        claimed_key = _bounded_token(execution_key_cid)
        if claimed_key and claimed_key != intent.execution_key_cid:
            raise XdistReuseCoordinationError(
                "publication execution key cid mismatch"
            )
        claimed_composite = _bounded_token(composite_phase_receipt_cid)
        if composite_phase_receipt_cid and not claimed_composite:
            raise XdistReuseCoordinationError(
                "composite phase receipt cid is not a public pin"
            )
        if claimed_composite and intent.composite_phase_receipt_cid != claimed_composite:
            raise XdistReuseCoordinationError(
                "composite phase receipt cid mismatch"
            )
    elif isinstance(receipt, TestPassReceipt):
        try:
            intent = ProofReusePublicationIntent.from_receipt(
                receipt,
                certificate=certificate,
                certificate_cid=certificate_cid,
                deferred_request=deferred_request,
                execution_key_cid=execution_key_cid,
                composite_phase_receipt_cid=composite_phase_receipt_cid,
            )
        except XdistReuseCoordinationError:
            raise
        except Exception as exc:
            raise XdistReuseCoordinationError(str(exc) or "publication intent invalid") from exc
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
        if execution_key_cid:
            payload["execution_key_cid"] = execution_key_cid
        if composite_phase_receipt_cid:
            payload["composite_phase_receipt_cid"] = composite_phase_receipt_cid
        if certificate is not None:
            payload["certificate"] = certificate
        if certificate_cid:
            payload["certificate_cid"] = certificate_cid
        if deferred_request is not None:
            payload["deferred_request"] = deferred_request
        try:
            intent = ProofReusePublicationIntent.from_dict(payload)
        except XdistReuseCoordinationError:
            raise
        except Exception as exc:
            raise XdistReuseCoordinationError(str(exc) or "publication intent invalid") from exc
    else:
        raise XdistReuseCoordinationError("publication receipt must be a mapping")

    attached_key = _attached_execution_key_cid(item)
    if attached_key and attached_key != intent.execution_key_cid:
        raise XdistReuseCoordinationError(
            "execution_key_cid disagrees with attached V2 key"
        )
    attached_composite = composite_phase_receipt_cid_from_item(item)
    if attached_composite:
        if (
            intent.composite_phase_receipt_cid
            and intent.composite_phase_receipt_cid != attached_composite
        ):
            raise XdistReuseCoordinationError(
                "composite phase receipt cid disagrees with attached receipt"
            )
        if not intent.composite_phase_receipt_cid:
            rebuilt = dict(intent.to_dict())
            rebuilt["composite_phase_receipt_cid"] = attached_composite
            rebuilt.pop("intent_id", None)
            intent = ProofReusePublicationIntent.from_dict(rebuilt)

    public_deferred = _public_deferred_request(intent.deferred_request)
    if intent.deferred_request is not None and public_deferred is None:
        raise XdistReuseCoordinationError("deferred request must be a public mapping")
    payload = intent.to_dict()
    _json_ready(payload)
    _bound_intent_size(payload)
    if not intent.execution_key_cid:
        raise XdistReuseCoordinationError("execution_key_cid pin is required")
    return intent


def queue_bounded_worker_intent(
    coordinator: ProofReuseXdistCoordinator,
    receipt: TestPassReceipt | Mapping[str, Any] | ProofReusePublicationIntent,
    **kwargs: Any,
) -> bool:
    """Enqueue one bounded public intent.  Workers still cannot write."""

    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        return False
    if not coordinator.can_accept_publication:
        return False
    for banned in (
        "private_key",
        "signing_key",
        "proving_key",
        "witness",
        "secret",
    ):
        kwargs.pop(banned, None)
    try:
        intent = bound_worker_intent(receipt, **kwargs)
    except Exception:
        coordinator.metrics.degraded(reason_code="publication_intent_invalid")
        return False
    return coordinator.queue_publication(intent)


def bound_worker_output(coordinator: ProofReuseXdistCoordinator) -> dict[str, Any]:
    """Return a frozen worker packet whose intents are public-only."""

    packet = coordinator.worker_output()
    if not isinstance(packet, dict):
        coordinator.mark_controller_unavailable()
        return coordinator.worker_output()
    if coordinator.role is ProofReuseXdistRole.WORKER and coordinator.can_write:
        coordinator.mark_controller_unavailable()
        return coordinator.worker_output()
    intents_payload = packet.get("intents")
    if not isinstance(intents_payload, list):
        coordinator.mark_controller_unavailable()
        return coordinator.worker_output()
    bounded: list[dict[str, Any]] = []
    try:
        for value in intents_payload:
            intent = bound_worker_intent(value)
            bounded.append(intent.to_dict())
    except Exception:
        coordinator.mark_controller_unavailable()
        unsigned = {
            "schema": packet.get("schema"),
            "interface": packet.get("interface"),
            "controller_id": packet.get("controller_id"),
            "session_id": packet.get("session_id"),
            "worker_id": packet.get("worker_id"),
            "worker_token": packet.get("worker_token"),
            "healthy": False,
            "metrics": coordinator.metrics.snapshot().to_dict(),
            "intents": [],
        }
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        unsigned["packet_id"] = hashlib.sha256(encoded).hexdigest()
        return json.loads(json.dumps(unsigned))
    packet["intents"] = bounded if coordinator.healthy else []
    unsigned = {key: value for key, value in packet.items() if key != "packet_id"}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    packet["packet_id"] = hashlib.sha256(encoded).hexdigest()
    return json.loads(json.dumps(packet))


def publish_accepted_reuse_evidence(
    coordinator: ProofReuseXdistCoordinator,
    store: Any,
    issuer: Any = None,
    *,
    candidate_store: Any = None,
) -> tuple[str, ...]:
    """Publish accepted intents.  Workers always return an empty tuple."""

    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        return ()
    if not controller_owns_publication(coordinator.role):
        coordinator.metrics.degraded(reason_code="worker_cannot_publish")
        return ()
    return coordinator.flush_publications(
        store,
        issuer,
        candidate_store=candidate_store,
    )


@dataclass(frozen=True)
class XdistReuseCoordinationResult:
    """Outcome of one bounded worker-to-controller handoff.

    Not a pytest test class.
    """

    __test__ = False

    queued: bool
    published: tuple[str, ...]
    role: str
    workers_may_publish: bool = False
    controller_owns_writes: bool = True
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    claim_class: str = CLAIM_CLASS
    interface: str = XDIST_REUSE_COORDINATION_RESULT_INTERFACE
    coordination_interface: str = XDIST_REUSE_COORDINATION_INTERFACE

    def __post_init__(self) -> None:
        if self.workers_may_publish:
            raise XdistReuseCoordinationError("workers must not publish")
        if self.may_authorize_skip:
            raise XdistReuseCoordinationError(
                "xdist reuse coordination must not authorize skip"
            )
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise XdistReuseCoordinationError(
                "xdist reuse coordination cannot admit production, "
                "self-approve, or change claims"
            )
        if self.role == ProofReuseXdistRole.WORKER.value and self.published:
            raise XdistReuseCoordinationError(
                "workers cannot publish accepted reuse evidence"
            )


class XdistReuseCoordinator:
    """PCTDD-030 adapter over ``ProofReuseXdistCoordinator``.

    Workers enqueue bounded public intents.  Only controller and standalone
    roles may flush accepted reuse evidence.
    """

    interface = XDIST_REUSE_COORDINATION_INTERFACE

    def __init__(self, inner: ProofReuseXdistCoordinator) -> None:
        if not isinstance(inner, ProofReuseXdistCoordinator):
            raise XdistReuseCoordinationError(
                "inner coordinator must be ProofReuseXdistCoordinator"
            )
        self._inner = inner

    @classmethod
    def wrap(cls, inner: ProofReuseXdistCoordinator) -> "XdistReuseCoordinator":
        return cls(inner)

    @classmethod
    def controller(cls, *, metrics: Any = None) -> "XdistReuseCoordinator":
        return cls(ProofReuseXdistCoordinator.controller(metrics=metrics))

    @classmethod
    def from_worker_input(
        cls,
        payload: Any,
        *,
        metrics: Any = None,
        worker_id: str = "",
    ) -> "XdistReuseCoordinator":
        return cls(
            ProofReuseXdistCoordinator.from_worker_input(
                payload,
                metrics=metrics,
                worker_id=worker_id,
            )
        )

    @property
    def inner(self) -> ProofReuseXdistCoordinator:
        return self._inner

    @property
    def role(self) -> ProofReuseXdistRole:
        return self._inner.role

    @property
    def healthy(self) -> bool:
        return self._inner.healthy

    @property
    def can_write(self) -> bool:
        return self._inner.can_write and controller_owns_publication(self.role)

    @property
    def can_publish(self) -> bool:
        return self.can_write

    @property
    def can_accept_publication(self) -> bool:
        return self._inner.can_accept_publication

    @property
    def pending_publications(self) -> int:
        return self._inner.pending_publications

    @property
    def metrics(self) -> Any:
        return self._inner.metrics

    def configure_worker(self, worker_id: str) -> dict[str, str]:
        return self._inner.configure_worker(worker_id)

    def queue_intent(
        self,
        receipt: TestPassReceipt | Mapping[str, Any] | ProofReusePublicationIntent,
        **kwargs: Any,
    ) -> bool:
        return queue_bounded_worker_intent(self._inner, receipt, **kwargs)

    def worker_output(self) -> dict[str, Any]:
        return bound_worker_output(self._inner)

    def accept_worker_output(self, payload: Any) -> bool:
        if self.role is not ProofReuseXdistRole.CONTROLLER:
            return False
        return self._inner.accept_worker_output(payload)

    def publish_accepted(
        self,
        store: Any,
        issuer: Any = None,
        *,
        candidate_store: Any = None,
    ) -> tuple[str, ...]:
        return publish_accepted_reuse_evidence(
            self._inner,
            store,
            issuer,
            candidate_store=candidate_store,
        )

    def mark_controller_unavailable(self, items: Any = ()) -> None:
        self._inner.mark_controller_unavailable(items)


__all__ = [
    "CLAIM_CLASS",
    "COORDINATION_DOES_NOT",
    "COORDINATION_ESTABLISHES",
    "COORDINATION_POLICY",
    "COORDINATION_UNAVAILABLE",
    "DEFAULT_POLICY_CID",
    "ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE",
    "ITEM_COMPOSITE_PHASE_RECEIPT_CID_ATTRIBUTE",
    "MAX_BOUNDED_INTENT_BYTES",
    "PREDECESSOR_INTERFACE",
    "XDIST_REUSE_COORDINATION_INTERFACE",
    "XDIST_REUSE_COORDINATION_POLICY_INTERFACE",
    "XDIST_REUSE_COORDINATION_RESULT_INTERFACE",
    "XDIST_REUSE_COORDINATION_SCHEMA",
    "XDIST_REUSE_INTENT_INTERFACE",
    "XdistReuseCoordinationError",
    "XdistReuseCoordinationResult",
    "XdistReuseCoordinator",
    "authority_descriptor",
    "bound_worker_intent",
    "bound_worker_output",
    "composite_phase_receipt_cid_from_item",
    "controller_owns_publication",
    "public_digest",
    "publish_accepted_reuse_evidence",
    "queue_bounded_worker_intent",
    "record_typed_unavailable",
    "typed_unavailable_records",
    "workers_may_publish",
]
