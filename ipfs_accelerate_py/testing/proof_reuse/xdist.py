"""Fail-closed pytest-xdist coordination for proof-backed test reuse.

Workers may read immutable candidates but never publish receipt or locator
state.  They return bounded publication intents to the one controller, which
deduplicates them and invokes the store's fenced publication entrypoint.
Missing, stale, or failed coordination disables proof skips and all writes;
ordinary pytest execution remains available.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ...agent_supervisor.proof.test_execution_contracts import (
    ReuseReasonCode,
    TestPassReceipt,
    TestProofCertificate,
    reuse_run,
)
from .lookup import ITEM_DECISION_ATTRIBUTE, SKIP_REASON_PREFIX
from .reporting import ProofReuseSessionMetrics

PROOF_REUSE_XDIST_INTERFACE: Final = "ProofReuseXdistCoordination@1"
PROOF_REUSE_XDIST_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-xdist@1"
)
WORKER_INPUT_KEY: Final = "ipfs_proof_reuse_coordination"
WORKER_OUTPUT_KEY: Final = "ipfs_proof_reuse_result"
COORDINATION_UNAVAILABLE: Final = "coordination_unavailable"
MAX_PUBLICATION_INTENTS: Final = 4096
MAX_PACKET_BYTES: Final = 8 * 1024 * 1024
MAX_RETAINED_PUBLIC_BYTES: Final = 2 * 1024 * 1024
MAX_PUBLIC_PIN_CHARS: Final = 256
MAX_PUBLIC_SCALAR_CHARS: Final = 4096
# Required V2 pins that must never be silently truncated on the wire.
REQUIRED_XDIST_CONTEXT_PIN_FIELDS: Final = (
    "receipt_cid",
    "execution_key_cid",
    "candidate_context_cid",
    "policy_cid",
    "statement_cid",
    "circuit_cid",
    "verifying_key_cid",
    "issuer_id",
    "epoch",
    "backend_id",
)
_RETAINED_PUBLIC_HEX_FIELDS: Final = frozenset(
    {
        "retained_receipt_bytes_hex",
        "retained_candidate_context_bytes_hex",
        "retained_execution_key_bytes_hex",
    }
)
_HEX_CHARACTERS: Final = frozenset("0123456789abcdefABCDEF")


class ProofReuseXdistRole(str, Enum):
    STANDALONE = "standalone"
    CONTROLLER = "controller"
    WORKER = "worker"


def _bounded_token(value: Any, *, length: int = 256) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value or len(value) > length:
        return ""
    if any(character.isspace() or ord(character) < 33 for character in value):
        return ""
    return value


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_deferred_request(value: Any) -> Mapping[str, Any] | None:
    """Strip witness/private fields before any worker/controller transport.

    Known deferred identity fields are preferred, but additional scalar public
    metadata (for example disagreement markers) may travel as long as no
    private/witness key or nested body is present.

    Present retained-hex or required pin fields that are oversized, malformed,
    or non-hex fail closed (``None``) rather than silently omitting the field.
    """

    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            return None
    if not isinstance(value, Mapping):
        return None
    # Prefer the receipt module's fail-closed public projection when the
    # payload looks like a deferred/controller context envelope.
    try:
        from .receipt import public_deferred_mapping

        known_markers = (
            "receipt_cid",
            "retained_receipt_bytes_hex",
            "retained_candidate_context_bytes_hex",
            "candidate_context_cid",
            "backend_id",
            "interface",
        )
        if any(marker in value for marker in known_markers):
            projected = public_deferred_mapping(value)
            if projected is None:
                return None
            # public_deferred_mapping only keeps known deferred fields; merge
            # additional scalar public metadata for disagreement markers etc.
            cleaned = dict(projected)
            private_markers = (
                "witness",
                "private",
                "secret",
                "password",
                "token",
                "credential",
                "api_key",
                "authorization",
                "cookie",
                "session",
            )
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if key in cleaned:
                    continue
                lowered = key.lower()
                if any(marker in lowered for marker in private_markers):
                    continue
                if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                    if isinstance(raw_value, str) and len(raw_value) > MAX_PUBLIC_SCALAR_CHARS:
                        return None
                    cleaned[key] = raw_value
            return cleaned or None
    except Exception:
        pass

    private_markers = (
        "witness",
        "private",
        "secret",
        "password",
        "token",
        "credential",
        "api_key",
        "authorization",
        "cookie",
        "session",
    )
    cleaned: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        lowered = key.lower()
        if any(marker in lowered for marker in private_markers):
            continue
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            if isinstance(raw_value, str):
                if key in _RETAINED_PUBLIC_HEX_FIELDS:
                    if not raw_value:
                        continue
                    if (
                        len(raw_value) % 2
                        or len(raw_value) > MAX_RETAINED_PUBLIC_BYTES * 2
                        or any(
                            character not in _HEX_CHARACTERS
                            for character in raw_value
                        )
                    ):
                        # Never silently drop a present retained-hex field.
                        return None
                    try:
                        data = bytes.fromhex(raw_value)
                    except ValueError:
                        return None
                    if len(data) > MAX_RETAINED_PUBLIC_BYTES:
                        return None
                    cleaned[key] = raw_value.lower()
                    continue
                if key in REQUIRED_XDIST_CONTEXT_PIN_FIELDS or key.endswith("_cid"):
                    if len(raw_value) > MAX_PUBLIC_PIN_CHARS:
                        return None
                elif len(raw_value) > MAX_PUBLIC_SCALAR_CHARS:
                    return None
            cleaned[key] = raw_value
        # Nested maps/lists are never transported: they may hide witness data.
    return cleaned or None


def _decode_retained_hex(value: str) -> bytes | None:
    if not value:
        return b""
    if (
        len(value) % 2
        or len(value) > MAX_RETAINED_PUBLIC_BYTES * 2
        or any(character not in _HEX_CHARACTERS for character in value)
    ):
        return None
    try:
        data = bytes.fromhex(value)
    except ValueError:
        return None
    if len(data) > MAX_RETAINED_PUBLIC_BYTES:
        return None
    return data


def reconstruct_controller_context_from_intent(
    intent: "ProofReusePublicationIntent",
    *,
    require_complete: bool = False,
) -> tuple[Any | None, str]:
    """Reconstruct controller-owned V2 context from one publication intent.

    Certificate fields on the intent never fill missing expected pins.  Shared
    by serial/direct-node and xdist controller flush paths.
    """

    try:
        from .candidate_publication import reconstruct_controller_owned_v2_context
        from .receipt import DeferredIssuanceEnvelope

        certificate = intent.certificate if isinstance(intent.certificate, Mapping) else None
        envelope = DeferredIssuanceEnvelope.from_mapping(intent.deferred_request)
        if envelope is None and intent.deferred_request is not None:
            return None, "deferred_request_malformed"
        if envelope is None:
            # Receipt-only intent: pins from the admitted receipt only.
            source = {
                "receipt_cid": intent.receipt_cid,
                "locator_cid": intent.locator_cid,
                "execution_key_cid": str(
                    intent.receipt.get("execution_key_cid") or ""
                ),
                "policy_cid": str(intent.receipt.get("policy_cid") or ""),
                "source": "xdist_receipt_only",
            }
            return reconstruct_controller_owned_v2_context(
                source,
                certificate=certificate,
                require_complete=require_complete,
            )
        retained_receipt = _decode_retained_hex(envelope.retained_receipt_bytes_hex)
        retained_candidate = _decode_retained_hex(
            envelope.retained_candidate_context_bytes_hex
        )
        retained_key = _decode_retained_hex(
            getattr(envelope, "retained_execution_key_bytes_hex", "") or ""
        )
        if (
            retained_receipt is None
            or retained_candidate is None
            or retained_key is None
        ):
            return None, "retained_public_bytes_invalid"
        return reconstruct_controller_owned_v2_context(
            envelope,
            retained_receipt_bytes=retained_receipt,
            retained_candidate_context_bytes=retained_candidate,
            retained_execution_key_bytes=retained_key,
            certificate=certificate,
            require_complete=require_complete,
        )
    except Exception:
        return None, "controller_context_intent_exception"


def _controller_deferred_request(
    intent: "ProofReusePublicationIntent",
) -> Any:
    """Reconstruct typed issuance input from retained public bytes.

    The worker carries only a bounded public envelope.  The controller opens
    and re-hashes its retained receipt/candidate bytes through the reviewed
    datasets request parser when available.  Certificate fields never fill
    missing expected pins.  A flattened mapping is only a non-authoritative
    fallback; the transaction's verifier will defer it.
    """

    try:
        from .receipt import (
            DeferredIssuanceEnvelope,
            reconstruct_deferred_request_from_public,
        )

        certificate = (
            intent.certificate if isinstance(intent.certificate, Mapping) else None
        )
        envelope = DeferredIssuanceEnvelope.from_mapping(intent.deferred_request)
        if envelope is None and intent.deferred_request is not None:
            # Malformed/oversized deferred payload: receipt-only fallback.
            return {
                "receipt_cid": intent.receipt_cid,
                "locator_cid": intent.locator_cid,
            }
        if envelope is None and intent.receipt_cid:
            envelope = DeferredIssuanceEnvelope(
                receipt_cid=intent.receipt_cid,
                locator_cid=intent.locator_cid,
                execution_key_cid=str(
                    intent.receipt.get("execution_key_cid") or ""
                ),
                policy_cid=str(intent.receipt.get("policy_cid") or ""),
                retained_receipt_bytes_hex="",
            )
        if envelope is None:
            return _public_deferred_request(intent.deferred_request)

        # Reconstruct controller-owned context first (rehash + no cert fill-in).
        context, context_reason = reconstruct_controller_context_from_intent(intent)
        del context_reason
        retained_receipt = _decode_retained_hex(envelope.retained_receipt_bytes_hex)
        retained_candidate = _decode_retained_hex(
            envelope.retained_candidate_context_bytes_hex
        )
        retained_key = _decode_retained_hex(
            getattr(envelope, "retained_execution_key_bytes_hex", "") or ""
        )
        if (
            retained_receipt is None
            or retained_candidate is None
            or retained_key is None
        ):
            return {
                "receipt_cid": intent.receipt_cid,
                "locator_cid": intent.locator_cid,
            }
        reconstructed = reconstruct_deferred_request_from_public(
            envelope,
            retained_receipt_bytes=retained_receipt,
            retained_candidate_context_bytes=retained_candidate,
            retained_execution_key_bytes=retained_key,
            certificate=certificate,
        )
        if reconstructed is None and context is not None:
            reconstructed = context.to_deferred_public_mapping()
        public = _public_deferred_request(
            reconstructed or envelope.to_dict()
        )
        if public is None:
            return {
                "receipt_cid": intent.receipt_cid,
                "locator_cid": intent.locator_cid,
            }
        # Prefer controller-reconstructed complete pins when available.
        if context is not None and context.is_complete:
            for name in REQUIRED_XDIST_CONTEXT_PIN_FIELDS:
                pin = context.pin_value(name)
                if pin:
                    public[name] = pin
            if context.proof_system_id:
                public["proof_system_id"] = context.proof_system_id
            if context.locator_cid:
                public["locator_cid"] = context.locator_cid
            if context.retained_receipt_bytes:
                public["retained_receipt_bytes_hex"] = (
                    context.retained_receipt_bytes_hex()
                )
            if context.retained_candidate_context_bytes:
                public["retained_candidate_context_bytes_hex"] = (
                    context.retained_candidate_context_bytes_hex()
                )
        try:
            from ipfs_datasets_py.logic.zkp.test_certificate_issuer import (
                DEFERRED_TEST_CERTIFICATE_REQUEST_INTERFACE,
                DeferredTestCertificateRequest,
            )

            typed_payload = dict(public)
            typed_payload["interface"] = (
                DEFERRED_TEST_CERTIFICATE_REQUEST_INTERFACE
            )
            typed = DeferredTestCertificateRequest.from_public_mapping(
                typed_payload
            )
            receipt_execution_key = str(
                intent.receipt.get("execution_key_cid") or ""
            )
            if (
                typed.receipt_cid != intent.receipt_cid
                or typed.execution_key_cid != receipt_execution_key
                or (
                    typed.locator_cid
                    and typed.locator_cid != intent.locator_cid
                )
            ):
                return public
            return typed
        except Exception:
            return public
    except Exception:
        return _public_deferred_request(intent.deferred_request)


@dataclass(frozen=True)
class ProofReusePublicationIntent:
    """Serializable, content-addressed unit submitted by one worker."""

    receipt: Mapping[str, Any]
    receipt_cid: str
    locator_cid: str
    certificate: Mapping[str, Any] | None = None
    certificate_cid: str = ""
    deferred_request: Mapping[str, Any] | None = None

    @classmethod
    def from_receipt(
        cls,
        receipt: TestPassReceipt,
        *,
        certificate: Mapping[str, Any] | None = None,
        certificate_cid: str = "",
        deferred_request: Mapping[str, Any] | None = None,
    ) -> "ProofReusePublicationIntent":
        if not isinstance(receipt, TestPassReceipt):
            raise TypeError("receipt must be TestPassReceipt")
        public_deferred = _public_deferred_request(deferred_request)
        if deferred_request is not None and public_deferred is None:
            # Present-but-invalid deferred context must not be silently dropped.
            raise ValueError("deferred request must be a public mapping")
        return cls(
            receipt=receipt.to_dict(),
            receipt_cid=receipt.receipt_id,
            locator_cid=receipt.locator_cid,
            certificate=dict(certificate) if certificate is not None else None,
            certificate_cid=_bounded_token(certificate_cid),
            deferred_request=public_deferred,
        ).validated()

    def validated(self) -> "ProofReusePublicationIntent":
        if not isinstance(self.receipt, Mapping):
            raise ValueError("publication receipt must be a mapping")
        typed = TestPassReceipt.from_dict(self.receipt)
        if not typed.admitted or not typed.all_phases_pass:
            raise ValueError("publication receipt is not admitted")
        if typed.receipt_id != self.receipt_cid:
            raise ValueError("publication receipt cid mismatch")
        if typed.locator_cid != self.locator_cid:
            raise ValueError("publication locator cid mismatch")
        if self.certificate is not None and not isinstance(
            self.certificate, Mapping
        ):
            raise ValueError("publication certificate must be a mapping")
        if self.certificate is None and self.certificate_cid:
            raise ValueError("certificate cid without certificate")
        if self.certificate is not None:
            typed_certificate = TestProofCertificate.from_dict(self.certificate)
            if typed_certificate.certificate_id != self.certificate_cid:
                raise ValueError("publication certificate cid mismatch")
            if typed_certificate.receipt_cid != typed.receipt_id:
                raise ValueError("publication certificate receipt mismatch")
            if typed_certificate.execution_key_cid != typed.execution_key_cid:
                raise ValueError("publication certificate execution key mismatch")
        public_deferred = _public_deferred_request(self.deferred_request)
        if self.deferred_request is not None and public_deferred is None:
            raise ValueError("deferred request must be a public mapping")
        if public_deferred is not self.deferred_request:
            object.__setattr__(self, "deferred_request", public_deferred)
        return self

    @property
    def intent_id(self) -> str:
        return _canonical_digest(
            {
                "receipt_cid": self.receipt_cid,
                "locator_cid": self.locator_cid,
                "certificate_cid": self.certificate_cid,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": dict(self.receipt),
            "receipt_cid": self.receipt_cid,
            "locator_cid": self.locator_cid,
            "certificate": (
                dict(self.certificate) if self.certificate is not None else None
            ),
            "certificate_cid": self.certificate_cid,
            "deferred_request": (
                dict(self.deferred_request)
                if self.deferred_request is not None
                else None
            ),
            "intent_id": self.intent_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofReusePublicationIntent":
        if not isinstance(payload, Mapping):
            raise ValueError("publication intent must be a mapping")
        allowed = {
            "receipt",
            "receipt_cid",
            "locator_cid",
            "certificate",
            "certificate_cid",
            "deferred_request",
            "intent_id",
        }
        if set(payload) - allowed:
            raise ValueError("publication intent contains unknown fields")
        intent = cls(
            receipt=payload.get("receipt") or {},
            receipt_cid=_bounded_token(payload.get("receipt_cid")),
            locator_cid=_bounded_token(payload.get("locator_cid")),
            certificate=payload.get("certificate"),
            certificate_cid=_bounded_token(payload.get("certificate_cid")),
            deferred_request=payload.get("deferred_request"),
        ).validated()
        claimed = _bounded_token(payload.get("intent_id"))
        if claimed and claimed != intent.intent_id:
            raise ValueError("publication intent identity mismatch")
        return intent


class ProofReuseXdistCoordinator:
    """Controller/worker state machine with one fenced write authority."""

    interface = PROOF_REUSE_XDIST_INTERFACE

    def __init__(
        self,
        role: ProofReuseXdistRole | str = ProofReuseXdistRole.STANDALONE,
        *,
        metrics: ProofReuseSessionMetrics | None = None,
        controller_id: str = "",
        session_id: str = "",
        worker_id: str = "",
        worker_token: str = "",
        healthy: bool = True,
        max_publication_intents: int = MAX_PUBLICATION_INTENTS,
    ) -> None:
        self.role = (
            role if isinstance(role, ProofReuseXdistRole) else ProofReuseXdistRole(role)
        )
        self.metrics = metrics or ProofReuseSessionMetrics()
        self.controller_id = _bounded_token(controller_id) or (
            f"controller:{uuid.uuid4().hex}"
            if self.role is not ProofReuseXdistRole.WORKER
            else ""
        )
        self.session_id = _bounded_token(session_id) or (
            uuid.uuid4().hex
            if self.role is not ProofReuseXdistRole.WORKER
            else ""
        )
        self.worker_id = _bounded_token(worker_id)
        self.worker_token = _bounded_token(worker_token)
        self.max_publication_intents = int(max_publication_intents)
        if self.max_publication_intents <= 0:
            raise ValueError("max_publication_intents must be positive")
        self._healthy = bool(healthy)
        if self.role is ProofReuseXdistRole.WORKER:
            self._healthy = bool(
                healthy
                and self.controller_id
                and self.session_id
                and self.worker_id
                and self.worker_token
            )
        self._worker_tokens: dict[str, str] = {}
        self._worker_packets: set[str] = set()
        self._worker_results: dict[str, str] = {}
        self._intents: dict[str, ProofReusePublicationIntent] = {}
        self._published: set[str] = set()
        self._worker_output_cache: dict[str, Any] | None = None
        self._degradation_recorded = False
        self._lock = threading.RLock()
        if not self._healthy:
            self._record_degraded(COORDINATION_UNAVAILABLE)

    @classmethod
    def controller(
        cls, *, metrics: ProofReuseSessionMetrics | None = None
    ) -> "ProofReuseXdistCoordinator":
        return cls(ProofReuseXdistRole.CONTROLLER, metrics=metrics)

    @classmethod
    def standalone(
        cls, *, metrics: ProofReuseSessionMetrics | None = None
    ) -> "ProofReuseXdistCoordinator":
        return cls(ProofReuseXdistRole.STANDALONE, metrics=metrics)

    @classmethod
    def from_worker_input(
        cls,
        payload: Any,
        *,
        metrics: ProofReuseSessionMetrics | None = None,
        worker_id: str = "",
    ) -> "ProofReuseXdistCoordinator":
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("missing controller payload")
            allowed = {
                "schema",
                "interface",
                "controller_id",
                "session_id",
                "worker_id",
                "worker_token",
            }
            if set(payload) - allowed:
                raise ValueError("unknown controller payload field")
            if payload.get("schema") != PROOF_REUSE_XDIST_SCHEMA:
                raise ValueError("controller schema mismatch")
            if payload.get("interface") != PROOF_REUSE_XDIST_INTERFACE:
                raise ValueError("controller interface mismatch")
            payload_worker_id = _bounded_token(payload.get("worker_id"))
            expected_worker_id = _bounded_token(worker_id)
            if expected_worker_id and payload_worker_id != expected_worker_id:
                raise ValueError("worker identity mismatch")
            return cls(
                ProofReuseXdistRole.WORKER,
                metrics=metrics,
                controller_id=_bounded_token(payload.get("controller_id")),
                session_id=_bounded_token(payload.get("session_id")),
                worker_id=payload_worker_id,
                worker_token=_bounded_token(payload.get("worker_token")),
                healthy=True,
            )
        except Exception:
            return cls(
                ProofReuseXdistRole.WORKER,
                metrics=metrics,
                worker_id=_bounded_token(worker_id),
                healthy=False,
            )

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    @property
    def can_skip(self) -> bool:
        return self.healthy

    @property
    def can_accept_publication(self) -> bool:
        return self.healthy

    @property
    def can_write(self) -> bool:
        return self.healthy and self.role in (
            ProofReuseXdistRole.STANDALONE,
            ProofReuseXdistRole.CONTROLLER,
        )

    @property
    def pending_publications(self) -> int:
        with self._lock:
            return len(self._intents)

    def _record_degraded(self, reason: str) -> None:
        if not self._degradation_recorded:
            self.metrics.degraded(reason_code=reason)
            self._degradation_recorded = True

    def mark_controller_unavailable(self, items: Iterable[Any] = ()) -> None:
        """Fence this coordinator and make already-marked items execute."""

        with self._lock:
            self._healthy = False
            self._intents.clear()
            self._record_degraded(COORDINATION_UNAVAILABLE)
        for item in items:
            force_real_execution(item)

    def configure_worker(self, worker_id: str) -> dict[str, str]:
        if self.role is not ProofReuseXdistRole.CONTROLLER or not self.healthy:
            raise RuntimeError(COORDINATION_UNAVAILABLE)
        safe_worker_id = _bounded_token(worker_id)
        if not safe_worker_id:
            raise ValueError("worker_id is required")
        with self._lock:
            token = uuid.uuid4().hex
            self._worker_tokens[safe_worker_id] = token
        return {
            "schema": PROOF_REUSE_XDIST_SCHEMA,
            "interface": PROOF_REUSE_XDIST_INTERFACE,
            "controller_id": self.controller_id,
            "session_id": self.session_id,
            "worker_id": safe_worker_id,
            "worker_token": token,
        }

    def queue_publication(
        self,
        intent_or_receipt: ProofReusePublicationIntent | TestPassReceipt,
        **kwargs: Any,
    ) -> bool:
        if not self.can_accept_publication:
            self._record_degraded(COORDINATION_UNAVAILABLE)
            return False
        try:
            intent = (
                intent_or_receipt
                if isinstance(intent_or_receipt, ProofReusePublicationIntent)
                else ProofReusePublicationIntent.from_receipt(
                    intent_or_receipt, **kwargs
                )
            ).validated()
        except Exception:
            self.metrics.degraded(reason_code="publication_intent_invalid")
            return False
        with self._lock:
            existing = self._intents.get(intent.intent_id)
            if existing is not None:
                if existing.to_dict() != intent.to_dict():
                    self.metrics.degraded(reason_code="publication_intent_disagrees")
                    self.mark_controller_unavailable()
                return False
            if intent.intent_id in self._published:
                return False
            if len(self._intents) >= self.max_publication_intents:
                self.metrics.degraded(reason_code="publication_over_budget")
                return False
            self._intents[intent.intent_id] = intent
        return True

    def worker_output(self) -> dict[str, Any]:
        with self._lock:
            if self._worker_output_cache is not None:
                return json.loads(json.dumps(self._worker_output_cache))
        packet = {
            "schema": PROOF_REUSE_XDIST_SCHEMA,
            "interface": PROOF_REUSE_XDIST_INTERFACE,
            "controller_id": self.controller_id,
            "session_id": self.session_id,
            "worker_id": self.worker_id,
            "worker_token": self.worker_token,
            "healthy": self.healthy,
            "metrics": self.metrics.snapshot().to_dict(),
            "intents": (
                [intent.to_dict() for intent in self._intents.values()]
                if self.healthy
                else []
            ),
        }
        try:
            packet["packet_id"] = _canonical_digest(packet)
            packet_size = len(
                json.dumps(
                    packet,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        except (TypeError, ValueError):
            self.mark_controller_unavailable()
            return self.worker_output()
        if packet_size > MAX_PACKET_BYTES:
            self.mark_controller_unavailable()
            return self.worker_output()
        with self._lock:
            self._worker_output_cache = packet
        return json.loads(json.dumps(packet))

    def accept_worker_output(self, payload: Any) -> bool:
        """Authenticate and merge one worker packet exactly once."""

        if self.role is not ProofReuseXdistRole.CONTROLLER or not self.healthy:
            return False
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("worker output must be a mapping")
            allowed = {
                "schema",
                "interface",
                "controller_id",
                "session_id",
                "worker_id",
                "worker_token",
                "healthy",
                "metrics",
                "intents",
                "packet_id",
            }
            if set(payload) - allowed:
                raise ValueError("unknown worker output field")
            encoded_size = len(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            if encoded_size > MAX_PACKET_BYTES:
                raise ValueError("worker output is over budget")
            if (
                payload.get("schema") != PROOF_REUSE_XDIST_SCHEMA
                or payload.get("interface") != PROOF_REUSE_XDIST_INTERFACE
                or payload.get("controller_id") != self.controller_id
                or payload.get("session_id") != self.session_id
            ):
                raise ValueError("worker output authority mismatch")
            worker_id = _bounded_token(payload.get("worker_id"))
            worker_token = _bounded_token(payload.get("worker_token"))
            if not worker_id or self._worker_tokens.get(worker_id) != worker_token:
                raise ValueError("worker output token mismatch")
            packet_id = _bounded_token(payload.get("packet_id"))
            if not packet_id:
                raise ValueError("worker packet identity missing")
            unsigned_payload = {
                key: value for key, value in payload.items() if key != "packet_id"
            }
            if packet_id != _canonical_digest(unsigned_payload):
                raise ValueError("worker packet content mismatch")
            with self._lock:
                if packet_id in self._worker_packets:
                    return True
                if worker_id in self._worker_results:
                    raise ValueError("worker submitted multiple result packets")
            intents_payload = payload.get("intents")
            if not isinstance(intents_payload, list):
                raise ValueError("worker intents must be a list")
            if len(intents_payload) > self.max_publication_intents:
                raise ValueError("too many worker intents")
            healthy = payload.get("healthy") is True
            parsed = (
                [ProofReusePublicationIntent.from_dict(value) for value in intents_payload]
                if healthy
                else []
            )
            self.metrics.merge(payload.get("metrics"), packet_id=packet_id)
            with self._lock:
                self._worker_packets.add(packet_id)
                self._worker_results[worker_id] = packet_id
            if not healthy:
                self.mark_controller_unavailable()
                return True
            for intent in parsed:
                self.queue_publication(intent)
            return True
        except Exception:
            self.metrics.degraded(reason_code="worker_output_rejected")
            return False

    def flush_publications(
        self,
        store: Any,
        issuer: Any = None,
        *,
        candidate_store: Any = None,
    ) -> tuple[str, ...]:
        """Publish each intent once through the controller's fenced store API.

        All certificate authority is delegated to
        ``ProofReuseControllerPublicationTransaction``.  Attached certificates
        and issuer-returned certificates therefore traverse the same artifact
        provenance and local cryptographic verification boundary before the
        transaction may index them.  If that transaction, issuer, provenance,
        or verifier is unavailable, this coordinator may retain only the
        receipt as non-authoritative retry state; it never falls back to a
        direct candidate-index write.  Workers supply only public deferred
        envelopes; private witness material is stripped before issuance.
        """

        if not self.can_write:
            self._record_degraded(COORDINATION_UNAVAILABLE)
            return ()
        published: list[str] = []
        with self._lock:
            intents = tuple(self._intents.values())

        # This transaction is the sole certificate-publication authority.  An
        # import/construction failure deliberately has no legacy issuer or
        # candidate-index fallback.
        transaction: Any = None
        try:
            from .publication import ProofReuseControllerPublicationTransaction

            transaction = ProofReuseControllerPublicationTransaction(
                store=store,
                candidate_store=candidate_store,
                issuer=issuer,
                owner_id=self.controller_id,
                metrics=self.metrics,
            )
        except Exception:
            transaction = None

        def retain_non_authoritative(
            intent: ProofReusePublicationIntent,
            *,
            already_retained: bool = False,
            reason_code: str,
            record_deferred: bool = True,
        ) -> bool:
            """Best-effort receipt retention with no candidate-index authority."""

            retained = bool(already_retained)
            if not retained:
                method = getattr(store, "put_receipt", None)
                if callable(method):
                    try:
                        result = method(intent.receipt)
                        stored = getattr(result, "stored", None)
                        if stored is None:
                            stored = bool(result)
                        retained = stored is True
                    except Exception:
                        retained = False
            if record_deferred:
                self.metrics.deferred(reason_code=reason_code)
            if not retained:
                self.metrics.degraded(
                    reason_code="cache_unavailable"
                )
                return False
            with self._lock:
                self._published.add(intent.intent_id)
                self._intents.pop(intent.intent_id, None)
            published.append(intent.intent_id)
            return True

        for intent in intents:
            if intent.intent_id in self._published:
                continue
            if transaction is None:
                retain_non_authoritative(
                    intent,
                    reason_code="deferred_issuer_unavailable",
                )
                continue

            request = _controller_deferred_request(intent)
            if request is None:
                request = {
                    "receipt_cid": intent.receipt_cid,
                    "locator_cid": intent.locator_cid,
                }
            try:
                outcome = transaction.publish_intent(
                    intent,
                    store=store,
                    candidate_store=candidate_store,
                    issuer=issuer,
                    deferred_request=request,
                )
            except Exception:
                # An incompatible/broken transaction is equivalent to an
                # unavailable authority boundary, never permission to issue or
                # index through legacy calls in this module.
                retain_non_authoritative(
                    intent,
                    reason_code="deferred_issuer_unavailable",
                )
                continue

            if (
                getattr(outcome, "published", False) is True
                and getattr(outcome, "indexed", False) is True
                and getattr(outcome, "put_candidate_called", False) is True
            ):
                with self._lock:
                    self._published.add(intent.intent_id)
                    self._intents.pop(intent.intent_id, None)
                published.append(intent.intent_id)
                continue

            if getattr(outcome, "put_candidate_called", False):
                # The sole authority transaction reached its atomic write and
                # that write failed or returned a non-indexed result.  Fence all
                # subsequent writes for this controller session.
                self.metrics.degraded(
                    reason_code=(
                        getattr(outcome, "reason_code", None)
                        or "publication_failed"
                    )
                )
                self.mark_controller_unavailable()
                break

            retain_non_authoritative(
                intent,
                already_retained=getattr(
                    outcome, "non_authoritative_retained", False
                ),
                reason_code=(
                    getattr(outcome, "reason_code", None)
                    or (
                        "issuer_unavailable"
                        if issuer is None
                        else "certificate_deferred"
                    )
                ),
                record_deferred=False,
            )
        return tuple(published)


def force_real_execution(item: Any) -> None:
    """Remove only proof-reuse skip markers and attach an explicit RUN decision."""

    markers = getattr(item, "own_markers", None)
    if isinstance(markers, list):
        retained = []
        for marker in markers:
            if getattr(marker, "name", "") != "skip":
                retained.append(marker)
                continue
            kwargs = getattr(marker, "kwargs", {})
            reason = kwargs.get("reason", "") if isinstance(kwargs, Mapping) else ""
            if not isinstance(reason, str) or not reason.startswith(SKIP_REASON_PREFIX):
                retained.append(marker)
        try:
            markers[:] = retained
        except Exception:
            pass
    try:
        setattr(
            item,
            ITEM_DECISION_ATTRIBUTE,
            reuse_run(ReuseReasonCode.COORDINATION_UNAVAILABLE),
        )
    except Exception:
        pass


__all__ = [
    "COORDINATION_UNAVAILABLE",
    "MAX_PACKET_BYTES",
    "MAX_PUBLICATION_INTENTS",
    "MAX_PUBLIC_PIN_CHARS",
    "MAX_PUBLIC_SCALAR_CHARS",
    "MAX_RETAINED_PUBLIC_BYTES",
    "PROOF_REUSE_XDIST_INTERFACE",
    "PROOF_REUSE_XDIST_SCHEMA",
    "REQUIRED_XDIST_CONTEXT_PIN_FIELDS",
    "WORKER_INPUT_KEY",
    "WORKER_OUTPUT_KEY",
    "ProofReusePublicationIntent",
    "ProofReuseXdistCoordinator",
    "ProofReuseXdistRole",
    "force_real_execution",
    "reconstruct_controller_context_from_intent",
]
