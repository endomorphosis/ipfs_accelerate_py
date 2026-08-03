"""Lazy datasets certificate verification adapter (``TestCertificateProvider@1``).

The shared pytest / proof-cache boundary needs a local verifier over retained
certificate bytes without making ``ipfs_datasets_py`` a cold-import dependency
and without ever proving during lookup.

Properties:

* Constructing the provider and inspecting capabilities never import datasets
  ZK backends, circuits, or issuers.
* Verification re-decodes exact retained canonical bytes, recomputes content
  identities, and pins public inputs from the caller's current policy — never
  from certificate self-description alone.
* The optional deferred issuer handle is never invoked by ``verify`` /
  ``lookup`` / retained-byte verification paths.
* Missing, incompatible, timeout, and ordinary exception states return
  RUN-compatible typed results (never implicit SKIP).
* Simulated / non-attested authority is always rejected.
"""

from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.test_execution_contracts import (
    CertificateAuthority,
    ProofBackendMode,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionContractError,
    TestPassReceipt,
    TestProofCertificate,
    decision_from_absence,
    decision_from_exception,
    reuse_run,
    reuse_skip,
)


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

TEST_CERTIFICATE_PROVIDER_INTERFACE: Final = "TestCertificateProvider@1"
TEST_CERTIFICATE_PROVIDER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-certificate-provider@1"
)
TEST_CERTIFICATE_VERIFICATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "test-certificate-verification-result@1"
)
TEST_CERTIFICATE_PROVIDER_VERSION: Final = 1

DEFAULT_VERIFIER_MODULE: Final = (
    "ipfs_datasets_py.logic.zkp.test_execution_certificate"
)
DEFAULT_BINDING_MODULE: Final = (
    "ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit"
)
DEFAULT_STATEMENT_MODULE: Final = (
    "ipfs_datasets_py.logic.zkp.statements.test_pass"
)
DEFAULT_ZKP_MODULE: Final = "ipfs_datasets_py.logic.zkp"

DEFAULT_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_MAX_BLOB_BYTES: Final = 1_048_576
DEFAULT_MAX_PROOF_BYTES: Final = 4 * 1024 * 1024
# PTR-153: public issued-material surface (certificate + proof + bindings).
ISSUED_TEST_CERTIFICATE_MATERIAL_INTERFACE: Final = "IssuedTestCertificateMaterial@1"
DEFAULT_MAX_ISSUED_MATERIAL_BYTES: Final = 4 * 1024 * 1024
DEFAULT_MAX_ISSUED_CERTIFICATE_BYTES: Final = 1_048_576

# Loader / interpreter injection must never reach a native child.
_NATIVE_INJECTION_ENV_PREFIXES: Final = ("LD_", "DYLD_", "PYTHON")
_NATIVE_INJECTION_ENV_KEYS: Final = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
    }
)
_PRIVATE_MATERIAL_MARKERS: Final = (
    "witness",
    "private",
    "secret",
    "opening",
    "proving_key",
    "receipt_opening",
    "retained_receipt",
    "local_witness",
    "private_axioms",
)

IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID: Final = (
    "ipfs_datasets_py.test_certificate"
)

_BACKEND_FROM_PROOF_SYSTEM: Final = {
    "groth16": "groth16",
    "g16": "groth16",
    "provekit": "provekit",
    "provekit-whir": "provekit",
    "whir": "provekit",
    "pk": "provekit",
}

# Map datasets CertificateVerificationReason values onto RUN-compatible codes.
_DATASETS_REASON_TO_REUSE: Final = MappingProxyType(
    {
        "verified": ReuseReasonCode.PROOF_CACHE_HIT,
        "malformed_certificate": ReuseReasonCode.MALFORMED_ARTIFACT,
        "malformed_proof": ReuseReasonCode.MALFORMED_ARTIFACT,
        "certificate_non_attested": ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        "unsupported_backend": ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        "backend_mismatch": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "backend_unavailable": ReuseReasonCode.VERIFIER_UNAVAILABLE,
        "backend_error": ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
        "proof_invalid": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "proof_digest_mismatch": ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        "proof_artifact_mismatch": ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        "circuit_mismatch": ReuseReasonCode.CIRCUIT_UNAVAILABLE,
        "verifying_key_mismatch": ReuseReasonCode.KEY_UNAVAILABLE,
        "statement_mismatch": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "receipt_mismatch": ReuseReasonCode.RECEIPT_MISMATCH,
        "execution_key_mismatch": ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        "issuer_mismatch": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "policy_mismatch": ReuseReasonCode.POLICY_MISMATCH,
        "public_inputs_mismatch": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "replay_detected": ReuseReasonCode.EXPIRED_OR_REVOKED,
    }
)

_SIMULATION_MARKERS: Final = (
    "demo",
    "fallback",
    "fake",
    "mock",
    "simulated",
    "simulation",
)

_IMPORT_LOCK: Final = threading.Lock()


# ---------------------------------------------------------------------------
# Errors / status
# ---------------------------------------------------------------------------


class TestCertificateProviderError(RuntimeError):
    """Raised only for programmer misuse of the adapter surface itself."""

    __test__ = False


class TestCertificateVerificationStatus(str, Enum):
    """High-level verification outcome."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"

    __test__ = False


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestCertificateVerificationResult:
    """RUN-compatible typed result for local certificate verification.

    Truthiness is intentionally disabled so callers must inspect
    :attr:`can_authorize_skip` / :attr:`verified` rather than treating the
    object as a boolean authority flag.
    """

    __test__: ClassVar[bool] = False

    status: TestCertificateVerificationStatus
    reason_code: ReuseReasonCode
    authority: CertificateAuthority = CertificateAuthority.NON_ATTESTED
    detail: str = ""
    certificate_cid: str = ""
    receipt_cid: str = ""
    backend_id: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, TestCertificateVerificationStatus)
            else TestCertificateVerificationStatus(self.status)
        )
        reason = (
            self.reason_code
            if isinstance(self.reason_code, ReuseReasonCode)
            else ReuseReasonCode(self.reason_code)
        )
        authority = (
            self.authority
            if isinstance(self.authority, CertificateAuthority)
            else CertificateAuthority(self.authority)
        )
        if status is TestCertificateVerificationStatus.VERIFIED:
            if reason is not ReuseReasonCode.PROOF_CACHE_HIT:
                raise ValueError("verified status requires proof_cache_hit reason")
            if authority is not CertificateAuthority.AUTHORITATIVE:
                raise ValueError("verified status requires authoritative authority")
        elif authority is CertificateAuthority.AUTHORITATIVE:
            raise ValueError("non-verified results cannot be authoritative")
        if reason is ReuseReasonCode.PROOF_CACHE_HIT and (
            status is not TestCertificateVerificationStatus.VERIFIED
        ):
            raise ValueError("proof_cache_hit is only valid for verified results")
        detail = self.detail if isinstance(self.detail, str) else str(self.detail)
        if len(detail) > 512:
            detail = detail[:512]
        diagnostics = dict(self.diagnostics or {})
        if len(diagnostics) > 32:
            raise ValueError("diagnostics exceed 32 keys")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        for name in ("certificate_cid", "receipt_cid", "backend_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                object.__setattr__(self, name, str(value) if value is not None else "")
            elif len(value) > 4_096:
                object.__setattr__(self, name, value[:4_096])

    def __bool__(self) -> bool:
        raise TypeError(
            "inspect .verified / .can_authorize_skip; "
            "verification results are not truthy"
        )

    @property
    def verified(self) -> bool:
        return self.status is TestCertificateVerificationStatus.VERIFIED

    @property
    def available(self) -> bool:
        return self.status is not TestCertificateVerificationStatus.UNAVAILABLE

    @property
    def authoritative(self) -> bool:
        return (
            self.verified
            and self.authority is CertificateAuthority.AUTHORITATIVE
        )

    @property
    def can_authorize_skip(self) -> bool:
        return self.authoritative

    @property
    def test_action(self) -> str:
        return "skip" if self.can_authorize_skip else "run"

    @property
    def interface(self) -> str:
        return TEST_CERTIFICATE_PROVIDER_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEST_CERTIFICATE_VERIFICATION_RESULT_SCHEMA,
            "contract_version": TEST_CERTIFICATE_PROVIDER_VERSION,
            "interface": TEST_CERTIFICATE_PROVIDER_INTERFACE,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "authority": self.authority.value,
            "verified": self.verified,
            "available": self.available,
            "authoritative": self.authoritative,
            "can_authorize_skip": self.can_authorize_skip,
            "test_action": self.test_action,
            "detail": self.detail,
            "certificate_cid": self.certificate_cid,
            "receipt_cid": self.receipt_cid,
            "backend_id": self.backend_id,
            "diagnostics": dict(self.diagnostics),
        }

    def to_reuse_decision(self) -> ReuseDecision:
        """Project this verification result onto a plugin ``ReuseDecision``."""

        if self.can_authorize_skip:
            return reuse_skip(
                certificate_cid=self.certificate_cid,
                receipt_cid=self.receipt_cid,
                reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
                authority=CertificateAuthority.AUTHORITATIVE,
                diagnostics=dict(self.diagnostics),
            )
        return reuse_run(
            self.reason_code,
            diagnostics={
                **dict(self.diagnostics),
                "verification_status": self.status.value,
                "detail": self.detail[:256],
            },
            authority=self.authority,
        )


@dataclass(frozen=True, slots=True)
class TestCertificateProviderCapability:
    """Cold capability declaration; never loads the optional backend."""

    __test__: ClassVar[bool] = False

    provider_id: str = IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID
    interface: str = TEST_CERTIFICATE_PROVIDER_INTERFACE
    version: int = TEST_CERTIFICATE_PROVIDER_VERSION
    enabled: bool = True
    imported: bool = False
    lazy: bool = True
    verification: bool = True
    issuance: bool = False
    prove_on_lookup: bool = False
    side_effect_free_probe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEST_CERTIFICATE_PROVIDER_SCHEMA,
            "provider_id": self.provider_id,
            "interface": self.interface,
            "version": self.version,
            "enabled": self.enabled,
            "imported": self.imported,
            "lazy": True,
            "verification": self.verification,
            "issuance": self.issuance,
            "prove_on_lookup": False,
            "side_effect_free_probe": True,
            "test_action_when_unavailable": "run",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains_simulation_marker(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return any(marker in text for marker in _SIMULATION_MARKERS)


def _contains_private_material_marker(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_")
    return any(marker in text for marker in _PRIVATE_MATERIAL_MARKERS)


def redact_provider_diagnostics(detail: str, *, limit: int = 256) -> str:
    """Bound and scrub exception/detail text so secrets never enter receipts."""

    text = str(detail or "")
    if _contains_private_material_marker(text):
        return "redacted_private_or_sensitive_detail"
    return text[:limit]


def sanitize_native_child_environment(
    source: Mapping[str, str] | None,
    *,
    artifacts_root: str,
    binary_path: str = "",
    allowlist: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return a strict child env that overwrites the pinned artifacts root.

    Excludes ``LD_PRELOAD`` / ``DYLD_*`` / interpreter injection variables.
    """

    ambient = dict(source or {})
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "USER",
        "LOGNAME",
        "TERM",
    }
    if allowlist is not None:
        allowed = {str(item) for item in allowlist}
    cleaned: dict[str, str] = {}
    for key, value in ambient.items():
        name = str(key)
        upper = name.upper()
        if upper in _NATIVE_INJECTION_ENV_KEYS:
            continue
        if any(upper.startswith(prefix) for prefix in _NATIVE_INJECTION_ENV_PREFIXES):
            continue
        if name not in allowed and upper not in {a.upper() for a in allowed}:
            continue
        cleaned[name] = str(value)
    # Overwrite rather than inherit ambient artifact/binary pins.
    cleaned["GROTH16_BACKEND_ARTIFACTS_ROOT"] = str(artifacts_root)
    if binary_path:
        cleaned["IPFS_DATASETS_GROTH16_BINARY"] = str(binary_path)
    cleaned["IPFS_DATASETS_ENABLE_GROTH16"] = "1"
    # Final guarantee against ambient re-injection.
    cleaned["GROTH16_BACKEND_ARTIFACTS_ROOT"] = str(artifacts_root)
    for banned in list(cleaned):
        upper = banned.upper()
        if upper in _NATIVE_INJECTION_ENV_KEYS or any(
            upper.startswith(prefix) for prefix in _NATIVE_INJECTION_ENV_PREFIXES
        ):
            cleaned.pop(banned, None)
    return cleaned


def admit_issued_certificate_material(
    material: Any,
    *,
    expected_circuit_cid: str = "",
    expected_verifying_key_cid: str = "",
    max_certificate_bytes: int = DEFAULT_MAX_ISSUED_CERTIFICATE_BYTES,
    max_proof_bytes: int = DEFAULT_MAX_PROOF_BYTES,
    max_material_bytes: int = DEFAULT_MAX_ISSUED_MATERIAL_BYTES,
) -> tuple[Mapping[str, Any] | None, str]:
    """Admit public issued material or return a reject reason (no authority).

    Malformed, oversized, provenance-mismatched, or structurally incomplete
    provider output is rejected.  Private witness / secret fields are refused.
    """

    if material is None:
        return None, "material_missing"

    def _has_private_keys(value: Any, *, depth: int = 0) -> bool:
        if depth > 12:
            return False
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _contains_private_material_marker(key):
                    return True
                if _has_private_keys(item, depth=depth + 1):
                    return True
            return False
        if isinstance(value, list):
            return any(_has_private_keys(item, depth=depth + 1) for item in value)
        return False

    public: dict[str, Any]
    if isinstance(material, Mapping):
        if _has_private_keys(material):
            return None, "private_material_present"
        public = dict(material)
    else:
        to_public = getattr(material, "to_public_dict", None)
        to_dict = getattr(material, "to_dict", None)
        payload: Any = None
        if callable(to_public):
            try:
                payload = to_public()
            except Exception:
                payload = None
        if payload is None and callable(to_dict):
            try:
                payload = to_dict(include_proof=True, include_ids=True)
            except TypeError:
                try:
                    payload = to_dict()
                except Exception:
                    payload = None
            except Exception:
                payload = None
        if isinstance(payload, Mapping):
            public = dict(payload)
        else:
            certificate = getattr(material, "certificate", None)
            cert_map: Mapping[str, Any] | None = None
            if isinstance(certificate, Mapping):
                cert_map = dict(certificate)
            elif certificate is not None and callable(getattr(certificate, "to_dict", None)):
                try:
                    cert_payload = certificate.to_dict(
                        include_proof=True, include_ids=True
                    )
                except TypeError:
                    try:
                        cert_payload = certificate.to_dict()
                    except Exception:
                        cert_payload = None
                except Exception:
                    cert_payload = None
                if isinstance(cert_payload, Mapping):
                    cert_map = dict(cert_payload)
            if cert_map is None:
                return None, "certificate_missing"
            public = {
                "interface": str(
                    getattr(material, "interface", "")
                    or ISSUED_TEST_CERTIFICATE_MATERIAL_INTERFACE
                ),
                "certificate": dict(cert_map),
                "proof_digest": str(getattr(material, "proof_digest", "") or ""),
                "proof_artifact_cid": str(
                    getattr(material, "proof_artifact_cid", "") or ""
                ),
                "circuit_cid": str(getattr(material, "circuit_cid", "") or ""),
                "verifying_key_cid": str(
                    getattr(material, "verifying_key_cid", "") or ""
                ),
                "proof_json": dict(getattr(material, "proof_json", {}) or {})
                if isinstance(getattr(material, "proof_json", None), Mapping)
                else {},
                "artifact_bindings": dict(
                    getattr(material, "artifact_bindings", {}) or {}
                )
                if isinstance(getattr(material, "artifact_bindings", None), Mapping)
                else {},
                "verified_locally": bool(
                    getattr(material, "verified_locally", True)
                ),
            }

    # Strip private keys from the public projection.
    def _strip_private(value: Any, *, depth: int = 0) -> Any:
        if depth > 12:
            return None
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if _contains_private_material_marker(key):
                    continue
                out[str(key)] = _strip_private(item, depth=depth + 1)
            return out
        if isinstance(value, list):
            return [_strip_private(item, depth=depth + 1) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (bytes, bytearray)):
            return f"bytes:{len(value)}"
        return str(type(value).__name__)

    public = _strip_private(public)
    if not isinstance(public, Mapping):
        return None, "material_malformed"

    certificate = public.get("certificate")
    if not isinstance(certificate, Mapping) or not certificate:
        return None, "certificate_missing"
    proof_digest = str(public.get("proof_digest") or certificate.get("proof_digest") or "")
    proof_artifact_cid = str(
        public.get("proof_artifact_cid")
        or certificate.get("proof_artifact_cid")
        or ""
    )
    circuit_cid = str(
        public.get("circuit_cid") or certificate.get("circuit_cid") or ""
    )
    verifying_key_cid = str(
        public.get("verifying_key_cid")
        or certificate.get("verifying_key_cid")
        or ""
    )
    if not proof_digest or not proof_artifact_cid:
        return None, "proof_identity_missing"
    if not circuit_cid or not verifying_key_cid:
        return None, "provenance_pins_missing"
    if expected_circuit_cid and circuit_cid != expected_circuit_cid:
        return None, "circuit_cid_provenance_mismatch"
    if expected_verifying_key_cid and verifying_key_cid != expected_verifying_key_cid:
        return None, "verifying_key_cid_provenance_mismatch"

    try:
        cert_bytes = json.dumps(
            dict(certificate),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        material_bytes = json.dumps(
            dict(public),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return None, "material_not_serializable"
    if len(cert_bytes) > max_certificate_bytes:
        return None, "certificate_oversized"
    if len(material_bytes) > max_material_bytes:
        return None, "material_oversized"
    proof_json = public.get("proof_json")
    if isinstance(proof_json, Mapping) and proof_json:
        try:
            proof_bytes = json.dumps(
                dict(proof_json),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            return None, "proof_malformed"
        if len(proof_bytes) > max_proof_bytes:
            return None, "proof_oversized"

    encoded_lower = material_bytes.decode("utf-8", errors="replace").lower()
    for marker in (
        "receipt_opening_hex",
        "private_axioms",
        "proving_key",
        "local_witness",
        "retained_receipt_bytes",
    ):
        if marker in encoded_lower:
            return None, "private_material_present"

    admitted = {
        "interface": str(
            public.get("interface") or ISSUED_TEST_CERTIFICATE_MATERIAL_INTERFACE
        ),
        "certificate": dict(certificate),
        "proof_digest": proof_digest,
        "proof_artifact_cid": proof_artifact_cid,
        "circuit_cid": circuit_cid,
        "verifying_key_cid": verifying_key_cid,
        "proof_json": dict(proof_json) if isinstance(proof_json, Mapping) else {},
        "artifact_bindings": dict(public.get("artifact_bindings") or {})
        if isinstance(public.get("artifact_bindings"), Mapping)
        else {},
        "verified_locally": bool(public.get("verified_locally", True)),
        "can_authorize_skip": False,
        "authority": "non_authoritative_until_controller_verify",
    }
    return MappingProxyType(admitted), ""


def _result(
    status: TestCertificateVerificationStatus,
    reason_code: ReuseReasonCode,
    *,
    authority: CertificateAuthority = CertificateAuthority.NON_ATTESTED,
    detail: str = "",
    certificate_cid: str = "",
    receipt_cid: str = "",
    backend_id: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> TestCertificateVerificationResult:
    return TestCertificateVerificationResult(
        status=status,
        reason_code=reason_code,
        authority=authority,
        detail=detail,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        backend_id=backend_id,
        diagnostics=dict(diagnostics or {}),
    )


def _unavailable(
    reason_code: ReuseReasonCode,
    detail: str,
    *,
    certificate_cid: str = "",
    receipt_cid: str = "",
    backend_id: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> TestCertificateVerificationResult:
    return _result(
        TestCertificateVerificationStatus.UNAVAILABLE,
        reason_code,
        detail=detail,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        backend_id=backend_id,
        diagnostics=diagnostics,
    )


def _rejected(
    reason_code: ReuseReasonCode,
    detail: str,
    *,
    certificate_cid: str = "",
    receipt_cid: str = "",
    backend_id: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> TestCertificateVerificationResult:
    return _result(
        TestCertificateVerificationStatus.REJECTED,
        reason_code,
        detail=detail,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        backend_id=backend_id,
        diagnostics=diagnostics,
    )


def _verified(
    *,
    certificate_cid: str,
    receipt_cid: str,
    backend_id: str = "",
    detail: str = "proof and local certificate bindings verified",
    diagnostics: Mapping[str, Any] | None = None,
) -> TestCertificateVerificationResult:
    return _result(
        TestCertificateVerificationStatus.VERIFIED,
        ReuseReasonCode.PROOF_CACHE_HIT,
        authority=CertificateAuthority.AUTHORITATIVE,
        detail=detail,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        backend_id=backend_id,
        diagnostics=diagnostics,
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _decode_retained_contract(
    raw: Any,
    contract_type: type[TestPassReceipt] | type[TestProofCertificate],
    *,
    max_blob_bytes: int,
    field_name: str,
) -> tuple[TestPassReceipt | TestProofCertificate | None, TestCertificateVerificationResult | None]:
    if type(raw) is not bytes:
        return None, _rejected(
            ReuseReasonCode.MALFORMED_ARTIFACT,
            f"{field_name} must be exact retained bytes",
        )
    if not raw or len(raw) > max_blob_bytes:
        reason = (
            ReuseReasonCode.OVER_BUDGET
            if len(raw) > max_blob_bytes
            else ReuseReasonCode.MALFORMED_ARTIFACT
        )
        return None, _rejected(reason, f"{field_name} is empty or over budget")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
        if not isinstance(payload, Mapping):
            return None, _rejected(
                ReuseReasonCode.MALFORMED_ARTIFACT,
                f"{field_name} must decode to a JSON object",
            )
        contract = contract_type.from_dict(payload)
    except TestExecutionContractError as exc:
        message = str(exc).lower()
        if "private" in message:
            return None, _rejected(
                ReuseReasonCode.PRIVATE_MATERIAL, f"{field_name} contains private material"
            )
        if "illegal-authority" in message or "simulated" in message:
            return None, _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED
                if "simulated" in message or "illegal-authority" in message
                else ReuseReasonCode.ILLEGAL_AUTHORITY,
                str(exc),
            )
        return None, _rejected(
            ReuseReasonCode.MALFORMED_ARTIFACT, f"{field_name} failed contract decode"
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, _rejected(
            ReuseReasonCode.MALFORMED_ARTIFACT,
            f"{field_name} is not canonical DAG-JSON: {type(exc).__name__}",
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open
        return None, _unavailable(
            ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
            f"{field_name} decode raised {type(exc).__name__}",
        )

    # Exact retained bytes are the immutable authority.
    if contract.canonical_bytes() != raw:
        return None, _rejected(
            ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
            f"{field_name} is not exact retained canonical form",
        )
    return contract, None


def _backend_id_from_proof_system(proof_system_id: str) -> str:
    text = str(proof_system_id or "").strip().lower().replace("_", "-")
    return _BACKEND_FROM_PROOF_SYSTEM.get(text, text)


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip().lower()


def _mapping_get(source: Mapping[str, Any] | None, *names: str, default: Any = None) -> Any:
    if not isinstance(source, Mapping):
        return default
    for name in names:
        if name in source:
            return source[name]
    return default


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def _datasets_reason_to_reuse(reason_value: str) -> ReuseReasonCode:
    text = str(reason_value or "").strip().lower()
    if text in _DATASETS_REASON_TO_REUSE:
        return _DATASETS_REASON_TO_REUSE[text]
    if _contains_simulation_marker(text):
        return ReuseReasonCode.CERTIFICATE_NON_ATTESTED
    return ReuseReasonCode.TRUST_POLICY_REJECTED


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class IpfsDatasetsTestCertificateProvider:
    """Bounded local verification adapter with an optional deferred issuer handle.

    Cold use (construction, capability inspection, issuer-handle access) never
    imports datasets ZK code.  Lookup/verification never calls ``prove`` or the
    issuer handle.
    """

    __test__ = False
    interface: ClassVar[str] = TEST_CERTIFICATE_PROVIDER_INTERFACE
    provider_id: ClassVar[str] = IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID
    version: ClassVar[int] = TEST_CERTIFICATE_PROVIDER_VERSION

    def __init__(
        self,
        *,
        importer: Callable[[str], Any] | None = None,
        backend: Any | None = None,
        issuer: Any | None = None,
        verify_fn: Callable[..., Any] | None = None,
        enabled: bool = True,
        verifier_module: str = DEFAULT_VERIFIER_MODULE,
        binding_module: str = DEFAULT_BINDING_MODULE,
        statement_module: str = DEFAULT_STATEMENT_MODULE,
        zkp_module: str = DEFAULT_ZKP_MODULE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_proof_bytes: int = DEFAULT_MAX_PROOF_BYTES,
    ) -> None:
        if importer is not None and not callable(importer):
            raise TestCertificateProviderError("importer must be callable")
        if verify_fn is not None and not callable(verify_fn):
            raise TestCertificateProviderError("verify_fn must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or float(timeout_seconds) <= 0
        ):
            raise TestCertificateProviderError("timeout_seconds must be positive")
        if (
            isinstance(max_blob_bytes, bool)
            or not isinstance(max_blob_bytes, int)
            or max_blob_bytes <= 0
        ):
            raise TestCertificateProviderError("max_blob_bytes must be a positive int")
        if (
            isinstance(max_proof_bytes, bool)
            or not isinstance(max_proof_bytes, int)
            or max_proof_bytes <= 0
        ):
            raise TestCertificateProviderError("max_proof_bytes must be a positive int")

        self._importer = importer or importlib.import_module
        self._backend = backend
        self._issuer = issuer
        self._verify_fn = verify_fn
        self._enabled = bool(enabled)
        self._verifier_module = str(verifier_module)
        self._binding_module = str(binding_module)
        self._statement_module = str(statement_module)
        self._zkp_module = str(zkp_module)
        self._timeout_seconds = float(timeout_seconds)
        self._max_blob_bytes = int(max_blob_bytes)
        self._max_proof_bytes = int(max_proof_bytes)
        self._loaded: dict[str, Any] | None = None
        self._import_attempted = False
        self._import_error: str | None = None
        self._prove_calls = 0  # observability for tests / audits
        self._verify_calls = 0

    # -- cold surface -------------------------------------------------------

    def capabilities(self) -> TestCertificateProviderCapability:
        """Return the local lazy declaration without importing datasets code."""

        return TestCertificateProviderCapability(
            enabled=self._enabled,
            imported=self._loaded is not None,
            issuance=self._issuer is not None,
            prove_on_lookup=False,
        )

    capability = capabilities

    @property
    def issuer_handle(self) -> Any | None:
        """Optional deferred issuer; never invoked by verification/lookup."""

        return self._issuer

    def get_issuer_handle(self) -> Any | None:
        """Compatibility spelling for :attr:`issuer_handle`."""

        return self._issuer

    @property
    def imported(self) -> bool:
        return self._loaded is not None

    @property
    def verify_call_count(self) -> int:
        return self._verify_calls

    @property
    def prove_call_count(self) -> int:
        return self._prove_calls

    # -- loading ------------------------------------------------------------

    def _load_datasets_surface(self) -> tuple[dict[str, Any] | None, TestCertificateVerificationResult | None]:
        if not self._enabled:
            return None, _unavailable(
                ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                "test certificate provider is disabled",
            )
        if self._loaded is not None:
            return self._loaded, None
        if self._verify_fn is not None:
            # Injected verifier: no datasets import required for the verify call
            # itself.  Binding construction still needs modules when not injected.
            self._loaded = {
                "verify_test_execution_certificate": self._verify_fn,
                "injected": True,
            }
            return self._loaded, None

        self._import_attempted = True
        with _IMPORT_LOCK:
            if self._loaded is not None:
                return self._loaded, None
            try:
                verifier_mod = self._importer(self._verifier_module)
                binding_mod = self._importer(self._binding_module)
                statement_mod = self._importer(self._statement_module)
                zkp_mod = self._importer(self._zkp_module)
            except TimeoutError:
                self._import_error = "TimeoutError"
                return None, _unavailable(
                    ReuseReasonCode.TIMEOUT,
                    "timed out importing datasets certificate verifier",
                )
            except Exception as exc:
                self._import_error = type(exc).__name__
                return None, _unavailable(
                    ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                    f"datasets certificate verifier unavailable: {type(exc).__name__}",
                    diagnostics={"exception_type": type(exc).__name__},
                )

            verify_fn = getattr(verifier_mod, "verify_test_execution_certificate", None)
            binding_cls = getattr(binding_mod, "TestPassCircuitBinding", None)
            certificate_cls = getattr(verifier_mod, "TestExecutionCertificate", None)
            zkp_proof_cls = getattr(zkp_mod, "ZKPProof", None)
            build_public_inputs = getattr(statement_mod, "build_public_inputs", None)
            build_statement = getattr(statement_mod, "build_statement", None)

            missing: list[str] = []
            if not callable(verify_fn):
                missing.append("verify_test_execution_certificate")
            if binding_cls is None:
                missing.append("TestPassCircuitBinding")
            if certificate_cls is None:
                missing.append("TestExecutionCertificate")
            if zkp_proof_cls is None:
                missing.append("ZKPProof")
            if not callable(build_public_inputs):
                missing.append("build_public_inputs")
            if not callable(build_statement):
                missing.append("build_statement")
            if missing:
                self._import_error = "incompatible:" + ",".join(missing)
                return None, _unavailable(
                    ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                    "datasets certificate surface is incompatible: "
                    + ", ".join(missing),
                    diagnostics={"missing": missing},
                )

            self._loaded = {
                "verify_test_execution_certificate": verify_fn,
                "TestPassCircuitBinding": binding_cls,
                "TestExecutionCertificate": certificate_cls,
                "ZKPProof": zkp_proof_cls,
                "build_public_inputs": build_public_inputs,
                "build_statement": build_statement,
                "injected": False,
            }
            return self._loaded, None

    # -- proof / binding extraction -----------------------------------------

    def _extract_proof(
        self,
        certificate: TestProofCertificate | Mapping[str, Any] | Any,
        requirements: Mapping[str, Any],
        surface: Mapping[str, Any],
    ) -> tuple[Any | None, TestCertificateVerificationResult | None]:
        proof = _mapping_get(requirements, "proof", "zkp_proof")
        if proof is None:
            proof = getattr(certificate, "proof", None)
        if proof is None:
            mapping = _as_mapping(certificate)
            if mapping is not None:
                proof = mapping.get("proof") or mapping.get("zkp_proof")
        proof_bytes = _mapping_get(requirements, "proof_bytes", "proof_data")
        if proof is None and proof_bytes is not None:
            if isinstance(proof_bytes, str):
                try:
                    proof_bytes = bytes.fromhex(proof_bytes)
                except ValueError:
                    return None, _rejected(
                        ReuseReasonCode.MALFORMED_ARTIFACT,
                        "proof_bytes must be raw bytes or hex",
                    )
            if type(proof_bytes) is not bytes:
                if isinstance(proof_bytes, (bytearray, memoryview)):
                    proof_bytes = bytes(proof_bytes)
                else:
                    return None, _rejected(
                        ReuseReasonCode.MALFORMED_ARTIFACT,
                        "proof_bytes must be exact bytes",
                    )
            if not proof_bytes or len(proof_bytes) > self._max_proof_bytes:
                return None, _rejected(
                    ReuseReasonCode.OVER_BUDGET
                    if len(proof_bytes) > self._max_proof_bytes
                    else ReuseReasonCode.MALFORMED_ARTIFACT,
                    "proof_bytes empty or over budget",
                )
            public_inputs = _mapping_get(
                requirements, "proof_public_inputs", "public_inputs", default={}
            )
            if not isinstance(public_inputs, Mapping):
                public_inputs = {}
            metadata = _mapping_get(requirements, "proof_metadata", default={})
            if not isinstance(metadata, Mapping):
                metadata = {}
            zkp_cls = surface.get("ZKPProof")
            if zkp_cls is None:
                proof = {
                    "proof_data": proof_bytes,
                    "public_inputs": dict(public_inputs),
                    "metadata": dict(metadata),
                    "timestamp": float(_mapping_get(requirements, "proof_timestamp", default=0) or 0),
                    "size_bytes": len(proof_bytes),
                }
            else:
                proof = zkp_cls(
                    proof_data=proof_bytes,
                    public_inputs=dict(public_inputs),
                    metadata=dict(metadata),
                    timestamp=float(
                        _mapping_get(requirements, "proof_timestamp", default=0) or 0
                    ),
                    size_bytes=len(proof_bytes),
                )
        if proof is None:
            return None, _rejected(
                ReuseReasonCode.MALFORMED_ARTIFACT,
                "verification requires retained proof bytes",
            )
        return proof, None

    def _pinned_public_inputs(
        self,
        receipt: TestPassReceipt,
        certificate: TestProofCertificate,
        requirements: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build verifier-pinned public inputs from receipt + current policy."""

        explicit = _mapping_get(
            requirements, "expected_public_inputs", "public_inputs"
        )
        if isinstance(explicit, Mapping) and explicit:
            # Caller-supplied pins win; certificate cannot override them later.
            return dict(explicit)

        policy_cid = str(
            _mapping_get(requirements, "policy_cid", default=certificate.policy_cid)
            or certificate.policy_cid
            or receipt.policy_cid
        )
        statement_cid = str(
            _mapping_get(
                requirements, "statement_cid", default=certificate.statement_cid
            )
            or certificate.statement_cid
        )
        circuit_cid = str(
            _mapping_get(requirements, "circuit_cid", default=certificate.circuit_cid)
            or certificate.circuit_cid
        )
        verifying_key_cid = str(
            _mapping_get(
                requirements,
                "verifying_key_cid",
                default=certificate.verifying_key_cid,
            )
            or certificate.verifying_key_cid
        )
        issuer_id = str(
            _mapping_get(requirements, "issuer_id", default=certificate.issuer_id)
            or certificate.issuer_id
        )
        # Prefer an explicit single trusted issuer when the policy supplies a set.
        trusted = _mapping_get(requirements, "trusted_issuer_ids")
        if isinstance(trusted, Sequence) and not isinstance(trusted, (str, bytes)):
            trusted_list = [str(item) for item in trusted if str(item).strip()]
            if trusted_list and issuer_id not in trusted_list:
                # Pin to policy; mismatch is detected later.
                pass
            if len(trusted_list) == 1 and not issuer_id:
                issuer_id = trusted_list[0]
        epoch = str(
            _mapping_get(requirements, "epoch", default=certificate.epoch)
            or certificate.epoch
        )
        allowed_epochs = _mapping_get(requirements, "allowed_epochs")
        if isinstance(allowed_epochs, Sequence) and not isinstance(
            allowed_epochs, (str, bytes)
        ):
            allowed_list = [str(item) for item in allowed_epochs if str(item).strip()]
            if len(allowed_list) == 1 and not epoch:
                epoch = allowed_list[0]

        proof_system_id = str(
            _mapping_get(
                requirements, "proof_system_id", default=certificate.proof_system_id
            )
            or certificate.proof_system_id
        )

        setup = _enum_value(receipt.setup_outcome) or "pass"
        call = _enum_value(receipt.call_outcome) or "pass"
        teardown = _enum_value(receipt.teardown_outcome) or "pass"

        inputs: dict[str, Any] = {
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": receipt.execution_key_cid,
            "policy_cid": policy_cid,
            "statement_cid": statement_cid,
            "circuit_cid": circuit_cid,
            "verifying_key_cid": verifying_key_cid,
            "proof_system_id": proof_system_id,
            "issuer_id": issuer_id,
            "issuer_key_id": receipt.issuer_key_id,
            "epoch": epoch,
            "setup_outcome": setup,
            "call_outcome": call,
            "teardown_outcome": teardown,
        }
        if receipt.locator_cid:
            inputs["locator_cid"] = receipt.locator_cid
        if receipt.completeness_receipt_cid:
            inputs.setdefault(
                "completeness_policy_cid", receipt.completeness_receipt_cid
            )
        completeness_policy = _mapping_get(
            requirements, "completeness_policy_cid", "runtime_completeness_policy"
        )
        if completeness_policy:
            inputs["completeness_policy_cid"] = str(completeness_policy)
        return inputs

    def _build_binding(
        self,
        receipt: TestPassReceipt,
        certificate: TestProofCertificate,
        requirements: Mapping[str, Any],
        surface: Mapping[str, Any],
    ) -> tuple[Any | None, TestCertificateVerificationResult | None]:
        supplied = _mapping_get(requirements, "binding")
        if supplied is not None:
            return supplied, None

        build_public_inputs = surface.get("build_public_inputs")
        build_statement = surface.get("build_statement")
        binding_cls = surface.get("TestPassCircuitBinding")
        if not callable(build_public_inputs) or not callable(build_statement):
            if surface.get("injected") and self._verify_fn is not None:
                # Injected verifiers may accept (certificate, binding=None) or a
                # custom binding object supplied by the test.
                return None, None
            return None, _unavailable(
                ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                "datasets binding surface is unavailable",
            )
        if binding_cls is None and not surface.get("injected"):
            return None, _unavailable(
                ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                "TestPassCircuitBinding is unavailable",
            )

        pinned = self._pinned_public_inputs(receipt, certificate, requirements)
        backend_id = str(
            _mapping_get(
                requirements,
                "backend_id",
                default=_backend_id_from_proof_system(
                    str(pinned.get("proof_system_id") or certificate.proof_system_id)
                ),
            )
            or _backend_id_from_proof_system(certificate.proof_system_id)
        )
        if _contains_simulation_marker(backend_id):
            return None, _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                "simulated backends cannot authorize verification",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
                backend_id=backend_id,
            )

        try:
            # Prefer the statement helpers when they accept the full pin map.
            try:
                public_inputs = build_public_inputs(**{
                    key: pinned[key]
                    for key in (
                        "receipt_cid",
                        "execution_key_cid",
                        "policy_cid",
                        "statement_cid",
                        "circuit_cid",
                        "verifying_key_cid",
                        "issuer_id",
                        "epoch",
                    )
                    if key in pinned
                })
                # Overlay outcome / optional pins not in the builder signature.
                if isinstance(public_inputs, Mapping):
                    merged = dict(public_inputs)
                    for key, value in pinned.items():
                        merged.setdefault(key, value)
                    public_inputs = merged
            except TypeError:
                public_inputs = pinned
            statement = build_statement(public_inputs)
            if binding_cls is None:
                return statement, None
            artifacts = _mapping_get(requirements, "verifier_artifacts", default={})
            if not isinstance(artifacts, Mapping):
                artifacts = {}
            binding = binding_cls(
                statement,
                backend_id=backend_id,
                proof_system_id=str(
                    pinned.get("proof_system_id") or certificate.proof_system_id
                ),
                circuit_cid=str(pinned.get("circuit_cid") or ""),
                verifying_key_cid=str(pinned.get("verifying_key_cid") or ""),
                statement_cid=str(pinned.get("statement_cid") or ""),
                issuer_id=str(pinned.get("issuer_id") or ""),
                policy_cid=str(pinned.get("policy_cid") or ""),
                epoch=str(pinned.get("epoch") or ""),
                verifier_artifacts=artifacts,
                replayed_certificate_ids=_mapping_get(
                    requirements, "replayed_certificate_ids", default=()
                )
                or (),
                replayed_proof_digests=_mapping_get(
                    requirements, "replayed_proof_digests", default=()
                )
                or (),
                replayed_tokens=_mapping_get(
                    requirements, "replayed_tokens", default=()
                )
                or (),
                max_proof_bytes=int(
                    _mapping_get(
                        requirements,
                        "max_proof_bytes",
                        default=self._max_proof_bytes,
                    )
                    or self._max_proof_bytes
                ),
            )
            return binding, None
        except Exception as exc:
            message = str(exc)
            if _contains_simulation_marker(message):
                return None, _rejected(
                    ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                    "binding rejected simulated authority",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
            return None, _rejected(
                ReuseReasonCode.TRUST_POLICY_REJECTED,
                f"could not build pinned verifier binding: {type(exc).__name__}",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
                diagnostics={"exception_type": type(exc).__name__},
            )

    def _reject_simulated_certificate(
        self,
        certificate: TestProofCertificate,
        receipt: TestPassReceipt,
    ) -> TestCertificateVerificationResult | None:
        mode = certificate.backend_mode
        authority = certificate.authority
        if mode is ProofBackendMode.SIMULATED:
            return _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                "simulated certificate authority is non-attested",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if authority is not CertificateAuthority.AUTHORITATIVE:
            return _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                "non-authoritative certificate cannot authorize skip",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if _contains_simulation_marker(certificate.proof_system_id):
            return _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                "simulated proof system is non-attested",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if _contains_simulation_marker(certificate.issuer_id):
            return _rejected(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                "simulated issuer is non-attested",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        metadata = certificate.metadata or {}
        for key, value in metadata.items():
            if _contains_simulation_marker(key) or _contains_simulation_marker(value):
                return _rejected(
                    ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                    "certificate metadata carries simulation markers",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
        return None

    def _bind_receipt_to_certificate(
        self,
        receipt: TestPassReceipt,
        certificate: TestProofCertificate,
        requirements: Mapping[str, Any],
    ) -> TestCertificateVerificationResult | None:
        if certificate.receipt_cid != receipt.receipt_id:
            return _rejected(
                ReuseReasonCode.RECEIPT_MISMATCH,
                "certificate receipt_cid does not match retained receipt",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if certificate.execution_key_cid != receipt.execution_key_cid:
            return _rejected(
                ReuseReasonCode.EXECUTION_KEY_MISMATCH,
                "certificate execution_key_cid does not match retained receipt",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        policy_cid = _mapping_get(requirements, "policy_cid")
        if policy_cid and str(policy_cid) not in {
            certificate.policy_cid,
            receipt.policy_cid,
        }:
            # Require the certificate to match the current policy pin when both set.
            if certificate.policy_cid and certificate.policy_cid != str(policy_cid):
                return _rejected(
                    ReuseReasonCode.POLICY_MISMATCH,
                    "certificate policy_cid does not match pinned policy",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
        for field_name, reason in (
            ("statement_cid", ReuseReasonCode.TRUST_POLICY_REJECTED),
            ("circuit_cid", ReuseReasonCode.CIRCUIT_UNAVAILABLE),
            ("verifying_key_cid", ReuseReasonCode.KEY_UNAVAILABLE),
            ("proof_system_id", ReuseReasonCode.TRUST_POLICY_REJECTED),
        ):
            pinned = _mapping_get(requirements, field_name)
            actual = getattr(certificate, field_name, "")
            if pinned and actual and str(pinned) != str(actual):
                return _rejected(
                    reason,
                    f"certificate {field_name} does not match pinned input",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
        trusted = _mapping_get(requirements, "trusted_issuer_ids")
        if isinstance(trusted, Sequence) and not isinstance(trusted, (str, bytes)):
            trusted_set = {str(item) for item in trusted if str(item).strip()}
            if trusted_set and certificate.issuer_id not in trusted_set:
                return _rejected(
                    ReuseReasonCode.TRUST_POLICY_REJECTED,
                    "certificate issuer is not in the pinned trust set",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
        allowed_epochs = _mapping_get(requirements, "allowed_epochs")
        if isinstance(allowed_epochs, Sequence) and not isinstance(
            allowed_epochs, (str, bytes)
        ):
            allowed = {str(item) for item in allowed_epochs if str(item).strip()}
            if allowed and certificate.epoch not in allowed:
                return _rejected(
                    ReuseReasonCode.EXPIRED_OR_REVOKED,
                    "certificate epoch is outside the pinned allow-list",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )
        return None

    def _map_datasets_result(
        self,
        raw: Any,
        *,
        certificate: TestProofCertificate,
        receipt: TestPassReceipt,
    ) -> TestCertificateVerificationResult:
        if raw is True:
            return _verified(
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if raw is False:
            return _rejected(
                ReuseReasonCode.TRUST_POLICY_REJECTED,
                "verifier rejected the certificate",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        if isinstance(raw, TestCertificateVerificationResult):
            return raw

        status_value = _enum_value(getattr(raw, "status", None) or _mapping_get(_as_mapping(raw), "status"))
        reason_value = _enum_value(
            getattr(raw, "reason", None)
            or getattr(raw, "reason_code", None)
            or _mapping_get(_as_mapping(raw), "reason", "reason_code")
        )
        detail = str(getattr(raw, "detail", "") or _mapping_get(_as_mapping(raw), "detail", default="") or "")
        backend_id = str(
            getattr(raw, "backend_id", "")
            or _mapping_get(_as_mapping(raw), "backend_id", default="")
            or ""
        )
        certificate_id = str(
            getattr(raw, "certificate_id", "")
            or _mapping_get(_as_mapping(raw), "certificate_id", default="")
            or certificate.certificate_id
        )
        can_skip = bool(getattr(raw, "can_authorize_skip", False))
        verified = bool(getattr(raw, "verified", False)) or status_value == "verified"

        if verified and can_skip and status_value in ("", "verified"):
            return _verified(
                certificate_cid=certificate_id,
                receipt_cid=receipt.receipt_id,
                backend_id=backend_id,
                detail=detail or "proof and local certificate bindings verified",
            )

        reason = _datasets_reason_to_reuse(reason_value)
        if reason is ReuseReasonCode.CERTIFICATE_NON_ATTESTED or _contains_simulation_marker(
            reason_value
        ) or _contains_simulation_marker(detail):
            reason = ReuseReasonCode.CERTIFICATE_NON_ATTESTED

        if status_value == "unavailable" or reason in {
            ReuseReasonCode.VERIFIER_UNAVAILABLE,
            ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
            ReuseReasonCode.KEY_UNAVAILABLE,
            ReuseReasonCode.CIRCUIT_UNAVAILABLE,
            ReuseReasonCode.TIMEOUT,
        }:
            # Key/circuit mismatches are rejections, not unavailability, when the
            # datasets reason is a mismatch.  Only pure unavailability maps here.
            if reason_value.endswith("_mismatch"):
                return _rejected(
                    reason,
                    detail or reason_value,
                    certificate_cid=certificate_id,
                    receipt_cid=receipt.receipt_id,
                    backend_id=backend_id,
                )
            if status_value == "unavailable" or reason_value in {
                "backend_unavailable",
                "unsupported_backend",
            }:
                mapped = (
                    ReuseReasonCode.VERIFIER_UNAVAILABLE
                    if reason_value == "backend_unavailable"
                    else (
                        ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
                        if reason_value == "unsupported_backend"
                        else reason
                    )
                )
                return _unavailable(
                    mapped,
                    detail or reason_value or "verifier unavailable",
                    certificate_cid=certificate_id,
                    receipt_cid=receipt.receipt_id,
                    backend_id=backend_id,
                )

        return _rejected(
            reason,
            detail or reason_value or "certificate verification rejected",
            certificate_cid=certificate_id,
            receipt_cid=receipt.receipt_id,
            backend_id=backend_id,
        )

    def _invoke_verifier(
        self,
        surface: Mapping[str, Any],
        certificate: TestProofCertificate,
        receipt: TestPassReceipt,
        requirements: Mapping[str, Any],
        proof: Any,
        binding: Any,
    ) -> TestCertificateVerificationResult:
        verify_fn = surface.get("verify_test_execution_certificate") or self._verify_fn
        if not callable(verify_fn):
            return _unavailable(
                ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                "certificate verifier function is unavailable",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )

        # Never touch the issuer / prove path during verification.
        if self._issuer is not None:
            for name in ("prove", "issue", "issue_certificate", "generate_proof"):
                method = getattr(self._issuer, name, None)
                if callable(method):
                    # Do not call; only count would happen if someone hooks it.
                    pass

        cert_cls = surface.get("TestExecutionCertificate")
        certificate_arg: Any = certificate
        if cert_cls is not None and not surface.get("injected"):
            try:
                from_tpc = getattr(cert_cls, "from_test_proof_certificate", None)
                if callable(from_tpc):
                    certificate_arg = from_tpc(certificate, proof)
                else:
                    certificate_arg = cert_cls.from_dict(
                        certificate.to_dict(), proof=proof
                    )
            except Exception as exc:
                return _rejected(
                    ReuseReasonCode.MALFORMED_ARTIFACT,
                    f"could not normalize certificate for verifier: {type(exc).__name__}",
                    certificate_cid=certificate.certificate_id,
                    receipt_cid=receipt.receipt_id,
                )

        backend = self._backend
        if backend is None:
            backend = _mapping_get(requirements, "backend")

        def _call() -> Any:
            # Prefer the datasets signature:
            # verify_test_execution_certificate(certificate, binding, backend=..., proof=...)
            try:
                return verify_fn(
                    certificate_arg,
                    binding,
                    backend,
                    proof=proof,
                )
            except TypeError:
                pass
            try:
                return verify_fn(
                    certificate_arg,
                    binding,
                    backend=backend,
                    proof=proof,
                )
            except TypeError:
                pass
            try:
                return verify_fn(certificate, receipt, requirements)
            except TypeError:
                return verify_fn(certificate_arg, binding)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                raw = future.result(timeout=self._timeout_seconds)
        except FuturesTimeoutError:
            return _unavailable(
                ReuseReasonCode.TIMEOUT,
                "certificate verification timed out",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        except TimeoutError:
            return _unavailable(
                ReuseReasonCode.TIMEOUT,
                "certificate verification timed out",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
            )
        except Exception as exc:
            return _unavailable(
                ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
                f"certificate verification raised {type(exc).__name__}",
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
                diagnostics={"exception_type": type(exc).__name__},
            )

        return self._map_datasets_result(
            raw, certificate=certificate, receipt=receipt
        )

    # -- public verification API --------------------------------------------

    def verify_retained_bytes(
        self,
        certificate_bytes: bytes,
        receipt_bytes: bytes,
        requirements: Mapping[str, Any] | None = None,
        *,
        proof: Any | None = None,
        proof_bytes: bytes | None = None,
        binding: Any | None = None,
    ) -> TestCertificateVerificationResult:
        """Verify using exact retained receipt and certificate canonical bytes."""

        self._verify_calls += 1
        requirements_map: dict[str, Any] = dict(requirements or {})
        if proof is not None:
            requirements_map.setdefault("proof", proof)
        if proof_bytes is not None:
            requirements_map.setdefault("proof_bytes", proof_bytes)
        if binding is not None:
            requirements_map.setdefault("binding", binding)

        certificate, cert_error = _decode_retained_contract(
            certificate_bytes,
            TestProofCertificate,
            max_blob_bytes=self._max_blob_bytes,
            field_name="certificate_bytes",
        )
        if cert_error is not None:
            return cert_error
        assert isinstance(certificate, TestProofCertificate)

        receipt, receipt_error = _decode_retained_contract(
            receipt_bytes,
            TestPassReceipt,
            max_blob_bytes=self._max_blob_bytes,
            field_name="receipt_bytes",
        )
        if receipt_error is not None:
            return receipt_error
        assert isinstance(receipt, TestPassReceipt)

        return self._verify_decoded(certificate, receipt, requirements_map)

    def verify_certificate(
        self,
        certificate: TestProofCertificate | Mapping[str, Any] | Any,
        receipt: TestPassReceipt | Mapping[str, Any] | Any,
        requirements: Mapping[str, Any] | None = None,
        *,
        proof: Any | None = None,
        binding: Any | None = None,
        certificate_bytes: bytes | None = None,
        receipt_bytes: bytes | None = None,
    ) -> TestCertificateVerificationResult:
        """Verify a certificate under pinned inputs; never invokes prove."""

        self._verify_calls += 1
        requirements_map: dict[str, Any] = dict(requirements or {})
        if proof is not None:
            requirements_map.setdefault("proof", proof)
        if binding is not None:
            requirements_map.setdefault("binding", binding)

        if certificate_bytes is not None or receipt_bytes is not None:
            if certificate_bytes is None or receipt_bytes is None:
                return _rejected(
                    ReuseReasonCode.MALFORMED_ARTIFACT,
                    "retained-byte verification requires both certificate and receipt bytes",
                )
            return self.verify_retained_bytes(
                certificate_bytes,
                receipt_bytes,
                requirements_map,
            )

        try:
            if isinstance(certificate, TestProofCertificate):
                cert_obj = certificate
            elif isinstance(certificate, Mapping):
                cert_obj = TestProofCertificate.from_dict(certificate)
            else:
                payload = _as_mapping(certificate)
                if payload is None:
                    return _rejected(
                        ReuseReasonCode.MALFORMED_ARTIFACT,
                        "certificate must be TestProofCertificate or mapping",
                    )
                cert_obj = TestProofCertificate.from_dict(payload)

            if isinstance(receipt, TestPassReceipt):
                receipt_obj = receipt
            elif isinstance(receipt, Mapping):
                receipt_obj = TestPassReceipt.from_dict(receipt)
            else:
                payload = _as_mapping(receipt)
                if payload is None:
                    return _rejected(
                        ReuseReasonCode.MALFORMED_ARTIFACT,
                        "receipt must be TestPassReceipt or mapping",
                    )
                receipt_obj = TestPassReceipt.from_dict(payload)
        except TestExecutionContractError as exc:
            message = str(exc).lower()
            if "simulated" in message or "illegal-authority" in message:
                return _rejected(
                    ReuseReasonCode.CERTIFICATE_NON_ATTESTED, str(exc)
                )
            return _rejected(ReuseReasonCode.MALFORMED_ARTIFACT, str(exc))
        except Exception as exc:
            return _unavailable(
                ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
                f"certificate/receipt normalization raised {type(exc).__name__}",
                diagnostics={"exception_type": type(exc).__name__},
            )

        # When objects are provided, still enforce that any supplied retained
        # identity matches recomputed content ids when present in requirements.
        claimed_cert_cid = _mapping_get(requirements_map, "certificate_cid")
        if claimed_cert_cid and str(claimed_cert_cid) != cert_obj.certificate_id:
            return _rejected(
                ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
                "certificate_cid does not match recomputed content identity",
                certificate_cid=cert_obj.certificate_id,
                receipt_cid=receipt_obj.receipt_id,
            )
        claimed_receipt_cid = _mapping_get(requirements_map, "receipt_cid")
        if claimed_receipt_cid and str(claimed_receipt_cid) != receipt_obj.receipt_id:
            return _rejected(
                ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
                "receipt_cid does not match recomputed content identity",
                certificate_cid=cert_obj.certificate_id,
                receipt_cid=receipt_obj.receipt_id,
            )

        return self._verify_decoded(cert_obj, receipt_obj, requirements_map)

    def verify(
        self,
        certificate: TestProofCertificate | Mapping[str, Any] | Any,
        receipt: TestPassReceipt | Mapping[str, Any] | Any,
        requirements: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestCertificateVerificationResult:
        """Primary verification entry point (``TestCertificateProvider@1``)."""

        return self.verify_certificate(
            certificate, receipt, requirements, **kwargs
        )

    def lookup(
        self,
        certificate: TestProofCertificate | Mapping[str, Any] | Any,
        receipt: TestPassReceipt | Mapping[str, Any] | Any,
        requirements: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestCertificateVerificationResult:
        """Lookup-path verification alias that never issues or proves."""

        # Explicitly refuse to call prove even if kwargs smuggle prove flags.
        kwargs.pop("prove", None)
        kwargs.pop("issue", None)
        kwargs.pop("issue_if_missing", None)
        return self.verify_certificate(
            certificate, receipt, requirements, **kwargs
        )

    def _verify_decoded(
        self,
        certificate: TestProofCertificate,
        receipt: TestPassReceipt,
        requirements: Mapping[str, Any],
    ) -> TestCertificateVerificationResult:
        simulated = self._reject_simulated_certificate(certificate, receipt)
        if simulated is not None:
            return simulated

        bound = self._bind_receipt_to_certificate(receipt, certificate, requirements)
        if bound is not None:
            return bound

        surface, load_error = self._load_datasets_surface()
        if load_error is not None:
            return load_error
        assert surface is not None

        proof, proof_error = self._extract_proof(certificate, requirements, surface)
        if proof_error is not None:
            return proof_error

        binding, binding_error = self._build_binding(
            receipt, certificate, requirements, surface
        )
        if binding_error is not None:
            return binding_error

        return self._invoke_verifier(
            surface,
            certificate,
            receipt,
            requirements,
            proof,
            binding,
        )

    def as_cache_verifier(
        self,
    ) -> Callable[
        [TestProofCertificate, TestPassReceipt, Mapping[str, Any]], bool
    ]:
        """Return a ``TestProofCache``-compatible verifier (exact ``True`` only)."""

        def _cache_verify(
            certificate: TestProofCertificate,
            receipt: TestPassReceipt,
            requirements: Mapping[str, Any],
        ) -> bool:
            result = self.verify(certificate, receipt, requirements)
            return True if result.can_authorize_skip else False

        return _cache_verify

    def prove(self, *args: Any, **kwargs: Any) -> None:
        """Issuance is not part of the lookup path.

        Explicit maintenance callers must use :attr:`issuer_handle` after the
        pass receipt is stored.  This method always refuses so accidental prove
        wiring cannot authorize reuse.
        """

        self._prove_calls += 1
        raise TestCertificateProviderError(
            "prove is not invoked by TestCertificateProvider lookup/verify; "
            "use the deferred issuer handle for explicit maintenance proving"
        )

    def admit_issued_material(
        self,
        material: Any,
        *,
        expected_circuit_cid: str = "",
        expected_verifying_key_cid: str = "",
    ) -> TestCertificateVerificationResult:
        """Admit public issued material for local inspection (PTR-153).

        Successful structural admission does **not** grant skip authority —
        the controller must still reverify under pinned context.  Malformed,
        oversized, provenance-mismatched, incomplete, or private-bearing
        material is rejected without authority.
        """

        admitted, reason = admit_issued_certificate_material(
            material,
            expected_circuit_cid=expected_circuit_cid,
            expected_verifying_key_cid=expected_verifying_key_cid,
            max_certificate_bytes=min(
                self._max_blob_bytes, DEFAULT_MAX_ISSUED_CERTIFICATE_BYTES
            ),
            max_proof_bytes=self._max_proof_bytes,
            max_material_bytes=DEFAULT_MAX_ISSUED_MATERIAL_BYTES,
        )
        if admitted is None:
            code = ReuseReasonCode.MALFORMED_ARTIFACT
            if reason in {
                "certificate_oversized",
                "proof_oversized",
                "material_oversized",
            }:
                code = ReuseReasonCode.MALFORMED_ARTIFACT
            elif "provenance" in reason or "mismatch" in reason:
                code = ReuseReasonCode.TRUST_POLICY_REJECTED
            elif reason in {"material_missing", "certificate_missing"}:
                code = ReuseReasonCode.MALFORMED_ARTIFACT
            if reason == "private_material_present":
                code = ReuseReasonCode.CERTIFICATE_NON_ATTESTED
            return _rejected(
                code,
                redact_provider_diagnostics(reason or "material_rejected"),
                diagnostics={
                    "admission": "rejected",
                    "reason": str(reason or "")[:96],
                    "can_authorize_skip": False,
                },
            )
        # Structural admission only — never skip authority from self-claims.
        return _result(
            TestCertificateVerificationStatus.REJECTED
            if not admitted.get("verified_locally")
            else TestCertificateVerificationStatus.UNAVAILABLE,
            ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
            if admitted.get("verified_locally")
            else ReuseReasonCode.TRUST_POLICY_REJECTED,
            authority=CertificateAuthority.NON_ATTESTED,
            detail="issued_material_admitted_pending_controller_verify",
            certificate_cid=str(
                (admitted.get("certificate") or {}).get("certificate_id")
                or (admitted.get("certificate") or {}).get("certificate_cid")
                or ""
            )[:128],
            receipt_cid=str(
                (admitted.get("certificate") or {}).get("receipt_cid")
                or (admitted.get("certificate") or {}).get("receipt_id")
                or ""
            )[:128],
            backend_id="groth16",
            diagnostics={
                "admission": "public_material_ok",
                "can_authorize_skip": False,
                "circuit_cid": str(admitted.get("circuit_cid") or "")[:128],
                "verifying_key_cid": str(
                    admitted.get("verifying_key_cid") or ""
                )[:128],
                "proof_digest": str(admitted.get("proof_digest") or "")[:96],
                "interface": str(admitted.get("interface") or "")[:96],
            },
        )


def inspect_test_certificate_provider_capability(
    *, enabled: bool = True, issuance: bool = False
) -> TestCertificateProviderCapability:
    """Pure capability inspection with no optional imports."""

    return TestCertificateProviderCapability(enabled=enabled, issuance=issuance)


__all__ = [
    "DEFAULT_BINDING_MODULE",
    "DEFAULT_MAX_BLOB_BYTES",
    "DEFAULT_MAX_ISSUED_CERTIFICATE_BYTES",
    "DEFAULT_MAX_ISSUED_MATERIAL_BYTES",
    "DEFAULT_MAX_PROOF_BYTES",
    "DEFAULT_STATEMENT_MODULE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_VERIFIER_MODULE",
    "IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID",
    "ISSUED_TEST_CERTIFICATE_MATERIAL_INTERFACE",
    "IpfsDatasetsTestCertificateProvider",
    "TEST_CERTIFICATE_PROVIDER_INTERFACE",
    "TEST_CERTIFICATE_PROVIDER_SCHEMA",
    "TEST_CERTIFICATE_PROVIDER_VERSION",
    "TEST_CERTIFICATE_VERIFICATION_RESULT_SCHEMA",
    "TestCertificateProviderCapability",
    "TestCertificateProviderError",
    "TestCertificateVerificationResult",
    "TestCertificateVerificationStatus",
    "admit_issued_certificate_material",
    "decision_from_absence",
    "decision_from_exception",
    "inspect_test_certificate_provider_capability",
    "redact_provider_diagnostics",
    "sanitize_native_child_environment",
]
