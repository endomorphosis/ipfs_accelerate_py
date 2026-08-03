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
            if isinstance(raw_value, str) and len(raw_value) > 4096:
                continue
            cleaned[key] = raw_value
        # Nested maps/lists are never transported: they may hide witness data.
    return cleaned or None


def _controller_deferred_request(
    intent: "ProofReusePublicationIntent",
) -> Mapping[str, Any] | None:
    """Reconstruct issuance input from public retained bytes, not worker trust."""

    try:
        from .receipt import (
            DeferredIssuanceEnvelope,
            reconstruct_deferred_request_from_public,
        )

        envelope = DeferredIssuanceEnvelope.from_mapping(intent.deferred_request)
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
        retained_receipt = None
        retained_candidate = None
        if envelope.retained_receipt_bytes_hex:
            try:
                retained_receipt = bytes.fromhex(
                    envelope.retained_receipt_bytes_hex
                )
            except ValueError:
                retained_receipt = None
        if envelope.retained_candidate_context_bytes_hex:
            try:
                retained_candidate = bytes.fromhex(
                    envelope.retained_candidate_context_bytes_hex
                )
            except ValueError:
                retained_candidate = None
        reconstructed = reconstruct_deferred_request_from_public(
            envelope,
            retained_receipt_bytes=retained_receipt,
            retained_candidate_context_bytes=retained_candidate,
        )
        return _public_deferred_request(reconstructed or envelope.to_dict())
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
        return cls(
            receipt=receipt.to_dict(),
            receipt_cid=receipt.receipt_id,
            locator_cid=receipt.locator_cid,
            certificate=dict(certificate) if certificate is not None else None,
            certificate_cid=_bounded_token(certificate_cid),
            deferred_request=_public_deferred_request(deferred_request),
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

        PTR-147: when a certificate is already attached, or when the issuer
        returns one, the complete candidate is published exactly once via
        ``put_candidate``.  A returned certificate is never discarded.
        ``put_receipt`` alone remains for deferred/no-certificate paths and
        never authorizes skip.  Workers supply only public deferred envelopes;
        private witness material is stripped before issuance.
        """

        if not self.can_write:
            self._record_degraded(COORDINATION_UNAVAILABLE)
            return ()
        published: list[str] = []
        with self._lock:
            intents = tuple(self._intents.values())

        # Prefer the atomic controller publication transaction when available.
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

        for intent in intents:
            if intent.intent_id in self._published:
                continue
            try:
                if intent.certificate is not None:
                    # Authoritative path: atomic put_candidate only.
                    method = getattr(store, "put_candidate", None)
                    if not callable(method):
                        raise TypeError("atomic candidate publisher unavailable")
                    result = method(
                        intent.receipt,
                        intent.certificate,
                        locator_cid=intent.locator_cid,
                        owner_id=self.controller_id,
                    )
                    stored = getattr(result, "stored", None) is True
                    indexed = getattr(result, "indexed", None) is True
                    if not (stored and indexed):
                        raise RuntimeError("atomic candidate publication rejected")
                    with self._lock:
                        self._published.add(intent.intent_id)
                        self._intents.pop(intent.intent_id, None)
                    published.append(intent.intent_id)
                    continue

                # No attached certificate: cold retain + issue + verify + publish.
                if transaction is not None and issuer is not None:
                    request = _controller_deferred_request(intent)
                    if request is None:
                        request = {
                            "receipt_cid": intent.receipt_cid,
                            "locator_cid": intent.locator_cid,
                        }
                    outcome = transaction.publish_intent(
                        intent,
                        store=store,
                        candidate_store=candidate_store,
                        issuer=issuer,
                        deferred_request=request,
                    )
                    if getattr(outcome, "published", False):
                        with self._lock:
                            self._published.add(intent.intent_id)
                            self._intents.pop(intent.intent_id, None)
                        published.append(intent.intent_id)
                        continue
                    # Deferred / failed issuance: retain non-authoritative
                    # receipt when possible; never partial skip authority.
                    if getattr(outcome, "put_candidate_called", False) and not getattr(
                        outcome, "published", False
                    ):
                        # put_candidate was attempted and rejected — fence.
                        raise RuntimeError(
                            getattr(outcome, "reason_code", None)
                            or "atomic candidate publication rejected"
                        )
                    method = getattr(store, "put_receipt", None)
                    if callable(method) and not getattr(
                        outcome, "non_authoritative_retained", False
                    ):
                        result = method(intent.receipt)
                        stored = getattr(result, "stored", None)
                        if stored is None:
                            stored = bool(result)
                        if stored is not True:
                            raise RuntimeError("receipt publication rejected")
                    with self._lock:
                        self._published.add(intent.intent_id)
                        self._intents.pop(intent.intent_id, None)
                    published.append(intent.intent_id)
                    continue

                # issuer is None or transaction unavailable: receipt-only path.
                method = getattr(store, "put_receipt", None)
                if not callable(method):
                    raise TypeError("receipt publisher unavailable")
                result = method(intent.receipt)
                stored = getattr(result, "stored", None)
                if stored is None:
                    stored = bool(result)
                if stored is not True:
                    raise RuntimeError("receipt publication rejected")
                with self._lock:
                    self._published.add(intent.intent_id)
                    self._intents.pop(intent.intent_id, None)
                published.append(intent.intent_id)
                if issuer is None:
                    self.metrics.deferred(reason_code="certificate_deferred")
                else:
                    # Legacy issue path without transaction: still must not
                    # discard a returned certificate.
                    try:
                        issue = getattr(issuer, "issue", None)
                        if not callable(issue):
                            raise TypeError("deferred issuer unavailable")
                        request = _controller_deferred_request(intent)
                        if request is None:
                            request = {
                                "receipt_cid": intent.receipt_cid,
                                "locator_cid": intent.locator_cid,
                            }
                        issue_result = issue(request)
                        status = str(
                            getattr(issue_result, "status", "")
                        ).lower()
                        certificate = getattr(issue_result, "certificate", None)
                        if certificate is None:
                            mapped = (
                                issue_result.to_dict()
                                if hasattr(issue_result, "to_dict")
                                else None
                            )
                            if isinstance(mapped, Mapping):
                                certificate = mapped.get("certificate")
                        if certificate is not None:
                            put_candidate = getattr(store, "put_candidate", None)
                            if not callable(put_candidate):
                                raise TypeError(
                                    "atomic candidate publisher unavailable"
                                )
                            cert_map = (
                                certificate
                                if isinstance(certificate, Mapping)
                                else (
                                    certificate.to_dict()
                                    if hasattr(certificate, "to_dict")
                                    else None
                                )
                            )
                            if not isinstance(cert_map, Mapping):
                                raise TypeError("certificate payload unavailable")
                            cand = put_candidate(
                                intent.receipt,
                                cert_map,
                                locator_cid=intent.locator_cid,
                                owner_id=self.controller_id,
                            )
                            stored = getattr(cand, "stored", None) is True
                            indexed = getattr(cand, "indexed", None) is True
                            if not (stored and indexed):
                                raise RuntimeError(
                                    "atomic candidate publication rejected"
                                )
                        elif status in {
                            "deferred",
                            "certificate_deferred",
                            "queued",
                        }:
                            self.metrics.deferred(
                                reason_code="certificate_deferred"
                            )
                        else:
                            self.metrics.deferred(
                                reason_code="certificate_deferred"
                            )
                    except Exception:
                        self.metrics.deferred(
                            reason_code="deferred_issuer_unavailable"
                        )
            except Exception:
                # Immutable orphan blobs are not authority.  Never attempt an
                # index fallback after an atomic candidate publication error.
                self.metrics.degraded(reason_code="publication_failed")
                self.mark_controller_unavailable()
                break
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
    "MAX_PUBLICATION_INTENTS",
    "PROOF_REUSE_XDIST_INTERFACE",
    "PROOF_REUSE_XDIST_SCHEMA",
    "WORKER_INPUT_KEY",
    "WORKER_OUTPUT_KEY",
    "ProofReusePublicationIntent",
    "ProofReuseXdistCoordinator",
    "ProofReuseXdistRole",
    "force_real_execution",
]
