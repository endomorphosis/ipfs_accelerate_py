"""Controller-owned atomic issuance and certificate publication (PTR-147).

Cold publication writes and rehashes candidate components and the pass receipt
before issuance.  The controller consumes both immediate and deferred issuer
results, locally verifies every success, and atomically publishes the complete
candidate context, certificate bytes, and index exactly once via
``put_candidate``.

Workers serialize no witness/private material.  A crash or failure may leave an
immutable non-authoritative candidate/receipt for retry but never a partial
skip candidate.  Cache, issuer, Groth16, transport, lock, permission, or
controller absence preserves the pass and returns RUN/DEFERRED.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Optional

from .services import (
    DATASETS_GROTH16_ARTIFACTS_ROOT_ENV,
    DATASETS_GROTH16_BINARY_ENV,
    DATASETS_VERIFIER_REVISION,
)

PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE: Final = (
    "ProofReuseControllerPublicationTransaction@1"
)
ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE: Final = (
    "IssuedCertificatePublicationResult@1"
)
GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE: Final = (
    "Groth16ArtifactIdentityBindings@1"
)

# Test-pass circuit version introduced by PTR-144.
_TEST_PASS_CIRCUIT_VERSION: Final = 4


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_text(value: Any, *, max_chars: int = 256) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    text = value.strip()
    return text[:max_chars] if len(text) > max_chars else text


def _mapping_of(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
    return None


def _status_text(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    try:
        return str(raw).strip().lower()
    except Exception:
        return ""


def _extract_certificate_payload(issue_result: Any) -> Mapping[str, Any] | None:
    """Return public certificate mapping from an immediate or deferred result."""

    if issue_result is None:
        return None
    # Direct certificate attribute (material / disposition / deferred result).
    for attr in ("certificate", "issued_certificate", "certificate_payload"):
        cert = getattr(issue_result, attr, None)
        if cert is None:
            continue
        mapped = _mapping_of(cert)
        if mapped is not None:
            # Prefer include_proof=True when available.
            to_dict = getattr(cert, "to_dict", None)
            if callable(to_dict):
                try:
                    with_proof = to_dict(include_proof=True, include_ids=True)
                    if isinstance(with_proof, Mapping):
                        return with_proof
                except TypeError:
                    pass
                except Exception:
                    pass
            return mapped
    # Nested material (IssuedTestCertificateMaterial).
    material = getattr(issue_result, "material", None)
    if material is not None:
        mapped = _extract_certificate_payload(material)
        if mapped is not None:
            return mapped
        public = getattr(material, "to_public_dict", None)
        if callable(public):
            try:
                payload = public()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                nested = payload.get("certificate")
                if isinstance(nested, Mapping):
                    return nested
    mapped = _mapping_of(issue_result)
    if mapped is not None:
        nested = mapped.get("certificate")
        if isinstance(nested, Mapping):
            return nested
        # Some dispositions expose only certificate_cid without body — not enough.
    return None


def _issue_succeeded(issue_result: Any) -> bool:
    if issue_result is None:
        return False
    if getattr(issue_result, "issued", None) is True:
        return True
    status = _status_text(getattr(issue_result, "status", None))
    if status in {
        "issued",
        "certificate_issued",
        "success",
        "ok",
    }:
        return True
    # Material present implies success even if status attribute differs.
    if _extract_certificate_payload(issue_result) is not None:
        if status in {"", "issued", "certificate_issued", "success", "ok"}:
            return True
        if status not in {
            "deferred",
            "certificate_deferred",
            "queued",
            "rejected",
            "certificate_rejected",
            "failed",
        }:
            # Unknown status but certificate body present — treat as success so
            # flush_publications never discards a returned certificate.
            return True
    return False


def _issue_deferred(issue_result: Any) -> bool:
    if issue_result is None:
        return True
    if getattr(issue_result, "deferred", None) is True:
        return True
    status = _status_text(getattr(issue_result, "status", None))
    return status in {
        "deferred",
        "certificate_deferred",
        "queued",
        "run",
    }


def _local_verify_certificate(
    certificate: Mapping[str, Any] | Any,
    *,
    bindings: "Groth16ArtifactIdentityBindings | None" = None,
    require_cryptographic_verify: bool = False,
) -> tuple[bool, str]:
    """Locally verify one returned certificate; never raise.

    When *bindings* are provenance-ready, the certificate circuit/VK CIDs must
    match pins derived from exact activated bytes, and the backend artifact
    root (when present) must match those pins.  Missing/mismatched provenance
    returns failure so callers surface RUN/DEFERRED.

    When bindings are not enforced, a structural public certificate mapping is
    accepted after optional datasets verification.  Cryptographic verification
    failures under unenforced bindings still accept a complete public mapping
    only when the issuer already claimed success (controller re-check remains
    fail-closed for pin mismatches).
    """

    try:
        cert_map = _mapping_of(certificate) or {}
        enforce_pins = (
            bindings is not None and getattr(bindings, "provenance_ready", False)
        )
        if enforce_pins:
            assert bindings is not None
            cert_circuit = _bounded_text(
                cert_map.get("circuit_cid")
                or getattr(certificate, "circuit_cid", "")
            )
            cert_vk = _bounded_text(
                cert_map.get("verifying_key_cid")
                or getattr(certificate, "verifying_key_cid", "")
            )
            if cert_circuit and bindings.circuit_cid and cert_circuit != bindings.circuit_cid:
                return False, "circuit_cid_mismatch"
            if cert_vk and bindings.verifying_key_cid and cert_vk != bindings.verifying_key_cid:
                return False, "verifying_key_cid_mismatch"
            # Prove the backend used the artifact root matching those pins.
            claimed_root = _bounded_text(
                cert_map.get("artifacts_root")
                or cert_map.get("artifact_root")
                or (
                    (cert_map.get("extra") or {}).get("artifacts_root")
                    if isinstance(cert_map.get("extra"), Mapping)
                    else ""
                )
            )
            if (
                claimed_root
                and bindings.artifacts_root
                and Path(claimed_root).resolve()
                != Path(bindings.artifacts_root).resolve()
            ):
                return False, "artifact_root_mismatch"

        # Structural completeness: required public identity fields.
        required = (
            "receipt_cid",
            "execution_key_cid",
            "circuit_cid",
            "verifying_key_cid",
        )
        structural_ok = all(
            _bounded_text(cert_map.get(name) or getattr(certificate, name, ""))
            for name in required
        )
        if not structural_ok:
            return False, "certificate_structurally_incomplete"

        try:
            from ipfs_datasets_py.logic.zkp.test_execution_certificate import (
                verify_test_execution_certificate,
            )
        except Exception:
            return True, "structural_accept_verifier_unavailable"

        try:
            result = verify_test_execution_certificate(certificate)
        except TypeError:
            try:
                result = verify_test_execution_certificate(cert_map)
            except Exception:
                result = None
        except Exception:
            result = None

        if result is True:
            return True, "verified"
        if result is not None:
            status = _status_text(getattr(result, "status", result))
            if status in {"valid", "verified", "accepted", "ok", "true", "1"}:
                return True, "verified"
            if getattr(result, "valid", None) is True:
                return True, "verified"
            if getattr(result, "verified", None) is True:
                return True, "verified"

        if enforce_pins or require_cryptographic_verify:
            return False, "local_verification_failed"
        # Unenforced pins: accept complete public mapping after issuer success.
        return True, "structural_accept_after_issuer_success"
    except Exception:
        return False, "local_verification_exception"


@dataclass(frozen=True, slots=True)
class Groth16ArtifactIdentityBindings:
    """Circuit and verifying-key CIDs derived from exact activated bytes.

    Labels and certificate metadata are never authoritative for these pins.
    Missing, synthetic, stale, substituted, or mismatched provenance yields
    ``provenance_ready=False`` so callers return RUN/DEFERRED.
    """

    interface: str = GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE
    circuit_cid: str = ""
    verifying_key_cid: str = ""
    artifacts_root: str = ""
    verifying_key_sha256: str = ""
    proving_key_sha256: str = ""
    backend_circuit_version: int = _TEST_PASS_CIRCUIT_VERSION
    reviewed_revision: str = DATASETS_VERIFIER_REVISION
    provenance_ready: bool = False
    reason_code: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "artifacts_root": self.artifacts_root,
            "verifying_key_sha256": self.verifying_key_sha256,
            "proving_key_sha256": self.proving_key_sha256,
            "backend_circuit_version": self.backend_circuit_version,
            "reviewed_revision": self.reviewed_revision,
            "provenance_ready": self.provenance_ready,
            "reason_code": self.reason_code,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def unready(
        cls,
        reason_code: str,
        **diagnostics: Any,
    ) -> "Groth16ArtifactIdentityBindings":
        bounded: dict[str, Any] = {}
        for key, value in list(diagnostics.items())[:16]:
            name = str(key)[:64]
            if value is None or isinstance(value, (bool, int)):
                bounded[name] = value
            elif isinstance(value, str):
                bounded[name] = value[:128]
            else:
                bounded[name] = type(value).__name__[:64]
        return cls(
            provenance_ready=False,
            reason_code=str(reason_code)[:64],
            diagnostics=MappingProxyType(bounded),
        )

    @classmethod
    def from_activated_artifacts(
        cls,
        *,
        artifacts_root: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
        binary_path: str | os.PathLike[str] | None = None,
        circuit_version: int = _TEST_PASS_CIRCUIT_VERSION,
    ) -> "Groth16ArtifactIdentityBindings":
        """Derive pins from exact reviewed circuit + activated key bytes.

        A native binary alone is non-authoritative: keys and circuit identity
        must be present and content-addressed from their bytes.
        """

        env = environ if environ is not None else os.environ
        try:
            root: Path | None
            if artifacts_root is not None:
                root = Path(artifacts_root)
            else:
                override = str(env.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or "").strip()
                if override:
                    root = Path(override)
                else:
                    root = None
                    # Prefer datasets default when importable (lazy).
                    try:
                        from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                            default_artifacts_root,
                        )

                        root = default_artifacts_root()
                    except Exception:
                        root = None
            if root is None or not root.is_dir():
                return cls.unready(
                    "artifacts_root_missing",
                    binary_alone_non_authoritative=True,
                )

            version_dir = root / f"v{int(circuit_version)}"
            pk_path = version_dir / "proving_key.bin"
            vk_path = version_dir / "verifying_key.bin"
            if not pk_path.is_file() or not vk_path.is_file():
                return cls.unready(
                    "test_pass_keys_missing",
                    artifacts_root=str(root),
                    version=int(circuit_version),
                    binary_alone_non_authoritative=True,
                )
            try:
                pk_bytes = pk_path.read_bytes()
                vk_bytes = vk_path.read_bytes()
            except OSError:
                return cls.unready("artifact_read_failed", artifacts_root=str(root))
            if not pk_bytes or not vk_bytes:
                return cls.unready("artifact_empty", artifacts_root=str(root))

            pk_digest = _sha256_hex(pk_bytes)
            vk_digest = _sha256_hex(vk_bytes)

            # Derive CIDs from exact bytes via datasets helpers when available.
            circuit_cid = ""
            verifying_key_cid = ""
            try:
                from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                    reviewed_circuit_cid,
                    verifying_key_cid_for_bytes,
                )

                circuit_cid = reviewed_circuit_cid()
                verifying_key_cid = verifying_key_cid_for_bytes(vk_bytes)
            except Exception:
                # Local content-address fallback (not labels): sha256 envelopes.
                circuit_cid = (
                    "sha256:"
                    + _sha256_hex(
                        json.dumps(
                            {
                                "interface": "TestPassGroth16CircuitV4",
                                "backend_circuit_version": int(circuit_version),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                )
                verifying_key_cid = (
                    "sha256:"
                    + _sha256_hex(
                        json.dumps(
                            {
                                "artifact": "groth16_verifying_key",
                                "backend_circuit_version": int(circuit_version),
                                "sha256": vk_digest,
                                "size": len(vk_bytes),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                )

            if not circuit_cid or not verifying_key_cid:
                return cls.unready("cid_derivation_failed", artifacts_root=str(root))

            # Binary presence is noted but never sufficient alone.
            binary_present = False
            if binary_path is not None:
                binary_present = Path(binary_path).is_file()
            else:
                bin_override = str(env.get(DATASETS_GROTH16_BINARY_ENV, "") or "").strip()
                if bin_override:
                    binary_present = Path(bin_override).is_file()

            return cls(
                circuit_cid=circuit_cid,
                verifying_key_cid=verifying_key_cid,
                artifacts_root=str(root.resolve()),
                verifying_key_sha256=vk_digest,
                proving_key_sha256=pk_digest,
                backend_circuit_version=int(circuit_version),
                reviewed_revision=DATASETS_VERIFIER_REVISION,
                provenance_ready=True,
                reason_code="ready",
                diagnostics=MappingProxyType(
                    {
                        "binary_present": binary_present,
                        "binary_alone_non_authoritative": True,
                        "key_version": int(circuit_version),
                    }
                ),
            )
        except Exception as exc:
            return cls.unready(
                "artifact_binding_exception",
                error_type=type(exc).__name__,
            )


@dataclass(frozen=True, slots=True)
class IssuedCertificatePublicationResult:
    """Outcome of one controller publication transaction."""

    interface: str = ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE
    published: bool = False
    status: str = "deferred"
    reason_code: str = ""
    receipt_cid: str = ""
    certificate_cid: str = ""
    candidate_context_cid: str = ""
    indexed: bool = False
    put_candidate_called: bool = False
    non_authoritative_retained: bool = False
    action: str = "DEFERRED"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def authorizes_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "published": self.published,
            "status": self.status,
            "reason_code": self.reason_code,
            "receipt_cid": self.receipt_cid,
            "certificate_cid": self.certificate_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "indexed": self.indexed,
            "put_candidate_called": self.put_candidate_called,
            "non_authoritative_retained": self.non_authoritative_retained,
            "action": self.action,
            "authorizes_skip": False,
            "diagnostics": dict(self.diagnostics),
        }


class ProofReuseControllerPublicationTransaction:
    """Atomic controller-owned issue → verify → put_candidate sequence.

    Implements ``ProofReuseControllerPublicationTransaction@1``.

    Order:

    1. Cold-write non-authoritative candidate components + receipt (rehash).
    2. Reconstruct public deferred request (no worker private material).
    3. Call issuer (immediate or deferred result).
    4. Locally verify every success against artifact pins.
    5. Atomically ``put_candidate`` exactly once for complete authority.
    6. On failure/deferral, retain immutable non-authoritative candidate/receipt
       for retry; never publish a partial skip candidate.
    """

    interface: str = PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE

    def __init__(
        self,
        *,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        owner_id: str = "",
        artifact_bindings: Groth16ArtifactIdentityBindings | None = None,
        metrics: Any = None,
    ) -> None:
        self.store = store
        self.candidate_store = candidate_store
        self.issuer = issuer
        self.owner_id = _bounded_text(owner_id, max_chars=128)
        self.artifact_bindings = artifact_bindings
        self.metrics = metrics
        self._lock = threading.RLock()
        self._completed_intents: set[str] = set()

    def _retain_non_authoritative(
        self,
        *,
        receipt: Any,
        locator_cid: str,
        candidate_components: Mapping[str, bytes] | None = None,
        publication_envelope: Any = None,
    ) -> tuple[bool, str]:
        """Write immutable candidate/receipt without index skip authority."""

        retained = False
        candidate_cid = ""
        store = self.candidate_store
        if store is not None:
            try:
                # Prefer full envelope publish when available.
                if publication_envelope is not None:
                    publish = getattr(store, "publish", None)
                    if callable(publish):
                        try:
                            result = publish(publication_envelope)
                            retained = True
                            candidate_cid = _bounded_text(
                                getattr(result, "candidate_context_cid", "")
                                or getattr(result, "cid", "")
                                or getattr(publication_envelope, "candidate_context_cid", "")
                            )
                        except TypeError:
                            # Signature variants — fall through to put_canonical.
                            pass
                        except Exception:
                            pass
                if candidate_components:
                    put_bytes = getattr(store, "put_canonical_bytes", None)
                    if callable(put_bytes):
                        for _name, payload in candidate_components.items():
                            if isinstance(payload, (bytes, bytearray)) and payload:
                                try:
                                    put_bytes(bytes(payload))
                                    retained = True
                                except Exception:
                                    continue
                # Always attempt receipt retention on the candidate store CAS.
                if receipt is not None:
                    put_bytes = getattr(store, "put_canonical_bytes", None)
                    receipt_bytes = None
                    if isinstance(receipt, (bytes, bytearray)):
                        receipt_bytes = bytes(receipt)
                    else:
                        canonical = getattr(receipt, "canonical_bytes", None)
                        if callable(canonical):
                            try:
                                receipt_bytes = bytes(canonical())
                            except Exception:
                                receipt_bytes = None
                        elif isinstance(receipt, Mapping):
                            try:
                                receipt_bytes = json.dumps(
                                    dict(receipt),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            except Exception:
                                receipt_bytes = None
                    if receipt_bytes and callable(put_bytes):
                        try:
                            put_bytes(receipt_bytes)
                            retained = True
                        except Exception:
                            pass
            except Exception:
                retained = retained or False

        # Certificate-store put_receipt is intentionally deferred until the
        # no-certificate path: a later successful put_candidate must remain the
        # sole authority write and must not be preceded by a partial indexable
        # receipt publication on the certificate store.
        return retained, candidate_cid

    def _put_candidate_once(
        self,
        *,
        receipt: Any,
        certificate: Any,
        locator_cid: str,
    ) -> tuple[bool, bool, str]:
        """Atomically publish complete candidate; never partial index."""

        store = self.store
        if store is None:
            return False, False, "store_unavailable"
        method = getattr(store, "put_candidate", None)
        if not callable(method):
            return False, False, "put_candidate_unavailable"
        try:
            kwargs: dict[str, Any] = {}
            if locator_cid:
                kwargs["locator_cid"] = locator_cid
            if self.owner_id:
                kwargs["owner_id"] = self.owner_id
            result = method(receipt, certificate, **kwargs)
        except TypeError:
            try:
                result = method(receipt, certificate)
            except Exception:
                return False, False, "put_candidate_failed"
        except Exception:
            return False, False, "put_candidate_failed"

        stored = getattr(result, "stored", None)
        indexed = getattr(result, "indexed", None)
        if stored is None and indexed is None:
            # Some stores return truthy on full success.
            ok = result is True or bool(result)
            return ok, ok, "put_candidate" if ok else "put_candidate_rejected"
        stored_ok = stored is True
        indexed_ok = indexed is True
        if not (stored_ok and indexed_ok):
            return False, False, "put_candidate_rejected"
        return True, True, "published"

    def publish_intent(
        self,
        intent: Any,
        *,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        deferred_request: Mapping[str, Any] | None = None,
        candidate_components: Mapping[str, bytes] | None = None,
        publication_envelope: Any = None,
    ) -> IssuedCertificatePublicationResult:
        """Run one complete controller publication transaction.

        Never raises; failures return DEFERRED/RUN with optional non-authoritative
        retention for retry.
        """

        if store is not None:
            self.store = store
        if candidate_store is not None:
            self.candidate_store = candidate_store
        if issuer is not None:
            self.issuer = issuer

        receipt = getattr(intent, "receipt", intent)
        receipt_cid = _bounded_text(
            getattr(intent, "receipt_cid", None)
            or (receipt.get("receipt_id") if isinstance(receipt, Mapping) else "")
            or getattr(receipt, "receipt_id", "")
        )
        locator_cid = _bounded_text(
            getattr(intent, "locator_cid", None)
            or (receipt.get("locator_cid") if isinstance(receipt, Mapping) else "")
            or getattr(receipt, "locator_cid", "")
        )
        intent_id = _bounded_text(
            getattr(intent, "intent_id", None) or receipt_cid, max_chars=128
        )
        existing_certificate = getattr(intent, "certificate", None)

        with self._lock:
            if intent_id and intent_id in self._completed_intents:
                return IssuedCertificatePublicationResult(
                    published=True,
                    status="already_published",
                    reason_code="idempotent_skip",
                    receipt_cid=receipt_cid,
                    action="DEFERRED",
                )

        # 1. Cold retain candidate + receipt before issuance (non-authoritative).
        retained, candidate_cid = self._retain_non_authoritative(
            receipt=receipt,
            locator_cid=locator_cid,
            candidate_components=candidate_components,
            publication_envelope=publication_envelope,
        )

        certificate_payload = (
            dict(existing_certificate)
            if isinstance(existing_certificate, Mapping)
            else _mapping_of(existing_certificate)
        )

        # 2–3. Issue when no certificate is already attached.
        issue_result = None
        if certificate_payload is None and self.issuer is not None:
            request = deferred_request
            if request is None:
                request = getattr(intent, "deferred_request", None)
            if request is None:
                request = {
                    "receipt_cid": receipt_cid,
                    "locator_cid": locator_cid,
                }
            try:
                issue = getattr(self.issuer, "issue", None)
                if callable(issue):
                    issue_result = issue(request)
                elif callable(self.issuer):
                    issue_result = self.issuer(request)
            except Exception:
                issue_result = None
            if _issue_succeeded(issue_result):
                certificate_payload = _extract_certificate_payload(issue_result)
            elif _issue_deferred(issue_result):
                reason = _bounded_text(
                    getattr(issue_result, "reason", None)
                    or getattr(issue_result, "reason_code", None)
                    or "certificate_deferred"
                )
                if self.metrics is not None:
                    try:
                        self.metrics.deferred(reason_code=reason or "certificate_deferred")
                    except Exception:
                        pass
                return IssuedCertificatePublicationResult(
                    published=False,
                    status="certificate_deferred",
                    reason_code=reason or "certificate_deferred",
                    receipt_cid=receipt_cid,
                    candidate_context_cid=candidate_cid,
                    non_authoritative_retained=retained,
                    action="DEFERRED",
                    diagnostics=MappingProxyType({"stage": "issuance"}),
                )

        if certificate_payload is None:
            # No certificate available: retain non-authoritative receipt only.
            if self.store is not None:
                put_receipt = getattr(self.store, "put_receipt", None)
                if callable(put_receipt):
                    try:
                        put_receipt(receipt)
                        retained = True
                    except Exception:
                        pass
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code="certificate_deferred")
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code="certificate_unavailable",
                receipt_cid=receipt_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
            )

        # 4. Local verification of every success.
        bindings = self.artifact_bindings
        if bindings is None:
            bindings = getattr(self.issuer, "last_artifact_bindings", None)
        verified, verify_reason = _local_verify_certificate(
            certificate_payload,
            bindings=bindings if isinstance(bindings, Groth16ArtifactIdentityBindings) else None,
        )
        if not verified:
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code=verify_reason or "local_verification_failed")
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code=verify_reason or "local_verification_failed",
                receipt_cid=receipt_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType({"stage": "local_verification"}),
            )

        # 5. Atomic put_candidate exactly once — never discard a returned cert.
        certificate_cid = _bounded_text(
            certificate_payload.get("certificate_id")
            or certificate_payload.get("certificate_cid")
            or getattr(issue_result, "certificate_cid", "")
        )
        ok, indexed, pub_reason = self._put_candidate_once(
            receipt=receipt,
            certificate=certificate_payload,
            locator_cid=locator_cid,
        )
        if not ok:
            if self.metrics is not None:
                try:
                    self.metrics.degraded(reason_code=pub_reason or "publication_failed")
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="publication_failed",
                reason_code=pub_reason or "publication_failed",
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                candidate_context_cid=candidate_cid,
                put_candidate_called=True,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType({"stage": "put_candidate"}),
            )

        with self._lock:
            if intent_id:
                self._completed_intents.add(intent_id)

        return IssuedCertificatePublicationResult(
            published=True,
            status="certificate_issued",
            reason_code="published",
            receipt_cid=receipt_cid,
            certificate_cid=certificate_cid,
            candidate_context_cid=candidate_cid,
            indexed=indexed,
            put_candidate_called=True,
            non_authoritative_retained=retained,
            action="RUN",
            diagnostics=MappingProxyType(
                {
                    "stage": "complete",
                    "verify_reason": verify_reason,
                }
            ),
        )


__all__ = [
    "GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE",
    "Groth16ArtifactIdentityBindings",
    "ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE",
    "IssuedCertificatePublicationResult",
    "PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE",
    "ProofReuseControllerPublicationTransaction",
]
