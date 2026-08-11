"""Canonical CIDv1 / IPLD / IPFS replication adapter (ASE-038).

Coordination manifests may only reference strict multiformats CIDv1 objects
under the frozen supervisor profile (base32, sha2-256, raw or dag-json).  This
module:

* computes the expected raw / DAG-JSON CID locally before trust;
* requires the underlying backend to return the same validated CID (or fails);
* re-fetches and rehashes bytes before admission into manifests;
* capability-gates CAR export (unsupported codecs/CAR fail closed);
* classifies ``ipfs_kit_py``, Kubo, and HuggingFace cache roles accurately;
* records explicit degradation (never silent fallback);
* bridges runtime-CAS / MCP++ hashes through :class:`IdentityLink` without
  treating those local IDs as coordination-epoch authority.

The HuggingFace cache adapter remains cache-only transport until conformant:
its synthetic ``bafy…`` keys are never admitted as IPLD CIDs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Optional, Tuple

from ipfs_accelerate_py import ipfs_backend_router
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    ALLOWED_CODECS,
    CID_BASE,
    CID_VERSION,
    DIGEST_SIZE,
    IDENTITY_LINK_SCHEMA,
    IdentityKind,
    IdentityLink,
    MH_TYPE,
    MultiformatsIdentityError,
    canonical_dag_json_bytes,
    cid_for_bytes,
    cid_for_dag_json,
    digest_hex_from_cid,
    link_payload_digest,
    link_raw_bytes,
    link_runtime_artifact,
    require_canonical_dag_json_bytes,
    validate_cid,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
VERIFIED_IPLD_SCHEMA: Final = f"{SCHEMA_PREFIX}/verified-ipld-backend@1"
BACKEND_CAPABILITY_RECEIPT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/backend-capability-receipt@1"
)
PUT_RECEIPT_SCHEMA: Final = f"{SCHEMA_PREFIX}/verified-ipld-put-receipt@1"
GET_RECEIPT_SCHEMA: Final = f"{SCHEMA_PREFIX}/verified-ipld-get-receipt@1"
ADMISSION_RECEIPT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/coordination-cid-admission@1"
)

COORDINATION_CODECS: Final = frozenset({"raw", "dag-json"})


class VerifiedIPLDError(ValueError):
    """Fail-closed error for non-conformant CID / codec / CAR / backend use."""


class BackendRoleName(str, Enum):
    """Mirror of router roles for receipt stability."""

    IPFS_KIT = "ipfs_kit_py"
    KUBO = "kubo"
    CACHE = "cache"
    UNKNOWN = "unknown"
    MEMORY = "memory"


# ---------------------------------------------------------------------------
# In-memory conformant store (hermetic tests + local verification)
# ---------------------------------------------------------------------------


class InMemoryConformantBackend:
    """Minimal IPFSBackend that emits real CIDv1 raw/sha2-256 identifiers.

    Used by tests and as a local reference store.  Role is reported as
    ``memory`` (not cache): identifiers are real multiformats CIDs.
    """

    backend_name = "memory"
    backend_role = ipfs_backend_router.ROLE_MEMORY

    def __init__(self) -> None:
        self._blocks: dict[str, bytes] = {}
        self._pins: set[str] = set()

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        if type(data) is not bytes:
            raise TypeError("data must be exact bytes")
        cid = cid_for_bytes(data, codec="raw")
        self._blocks[cid] = data
        if pin:
            self._pins.add(cid)
        return cid

    def cat(self, cid: str) -> bytes:
        if cid not in self._blocks:
            raise RuntimeError(f"CID not found: {cid}")
        return self._blocks[cid]

    def pin(self, cid: str) -> None:
        if cid not in self._blocks:
            raise RuntimeError(f"CID not found: {cid}")
        self._pins.add(cid)

    def unpin(self, cid: str) -> None:
        self._pins.discard(cid)

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        if codec not in ALLOWED_CODECS:
            raise RuntimeError(f"unsupported codec: {codec!r}")
        if type(data) is not bytes:
            raise TypeError("data must be exact bytes")
        cid = cid_for_bytes(data, codec=codec)
        self._blocks[cid] = data
        self._pins.add(cid)
        return cid

    def block_get(self, cid: str) -> bytes:
        return self.cat(cid)

    def add_path(
        self,
        path: str,
        *,
        recursive: bool = True,
        pin: bool = True,
        chunker: Optional[str] = None,
    ) -> str:
        from pathlib import Path

        _ = recursive, chunker
        return self.add_bytes(Path(path).read_bytes(), pin=pin)

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        from pathlib import Path

        Path(output_path).write_bytes(self.cat(cid))

    def ls(self, cid: str) -> list[str]:
        _ = cid
        return []

    def dag_export(self, cid: str) -> bytes:
        # Minimal single-block "CAR-like" envelope for capability tests only.
        # Real CAR encoding is out of scope; presence of bytes proves the gate
        # opens for conformant roles that claim CAR support.
        payload = self.cat(cid)
        header = b"verified-ipld-car-v1\n" + cid.encode("utf-8") + b"\n"
        return header + payload


# ---------------------------------------------------------------------------
# Capability and put/get receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapabilityReceipt:
    """Replication-facing capability matrix for the active backend.

    ``degraded`` is true whenever the preferred ``ipfs_kit_py`` path was not
    used, the role is cache-only, or explicit degradation reasons were
    recorded.  Cache roles always report ``conformant_cid=False`` and
    ``supports_car=False``.
    """

    schema: str
    backend_name: str
    role: str
    preferred_name: str
    preferred_available: bool
    degraded: bool
    degradation_reasons: Tuple[str, ...]
    conformant_cid: bool
    supports_raw: bool
    supports_dag_json: bool
    supports_car: bool
    supports_pin: bool
    codec_preservation_guaranteed: bool
    notes: Tuple[str, ...]
    candidate_order: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != BACKEND_CAPABILITY_RECEIPT_SCHEMA:
            raise VerifiedIPLDError("unsupported backend capability receipt schema")
        if self.role == BackendRoleName.CACHE.value:
            if self.conformant_cid:
                raise VerifiedIPLDError(
                    "cache role must not claim conformant_cid"
                )
            if self.supports_car:
                raise VerifiedIPLDError("cache role must not claim CAR support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend_name": self.backend_name,
            "role": self.role,
            "preferred_name": self.preferred_name,
            "preferred_available": self.preferred_available,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "conformant_cid": self.conformant_cid,
            "supports_raw": self.supports_raw,
            "supports_dag_json": self.supports_dag_json,
            "supports_car": self.supports_car,
            "supports_pin": self.supports_pin,
            "codec_preservation_guaranteed": self.codec_preservation_guaranteed,
            "notes": list(self.notes),
            "candidate_order": list(self.candidate_order),
        }

    def content_cid(self) -> str:
        return cid_for_dag_json(self.to_dict(), for_identity=True)

    @classmethod
    def from_selection(
        cls,
        selection: ipfs_backend_router.BackendSelectionReceipt,
    ) -> "BackendCapabilityReceipt":
        caps = selection.capabilities
        role = selection.selected_role
        # Normalize known router roles onto the receipt vocabulary.
        if role == ipfs_backend_router.ROLE_IPFS_KIT:
            role = BackendRoleName.IPFS_KIT.value
        elif role == ipfs_backend_router.ROLE_KUBO:
            role = BackendRoleName.KUBO.value
        elif role == ipfs_backend_router.ROLE_CACHE:
            role = BackendRoleName.CACHE.value
        elif role == ipfs_backend_router.ROLE_MEMORY:
            role = BackendRoleName.MEMORY.value
        else:
            role = BackendRoleName.UNKNOWN.value

        conformant = bool(caps.conformant_cid)
        supports_car = bool(caps.supports_car)
        if role == BackendRoleName.CACHE.value:
            conformant = False
            supports_car = False

        return cls(
            schema=BACKEND_CAPABILITY_RECEIPT_SCHEMA,
            backend_name=selection.selected_name,
            role=role,
            preferred_name=selection.preferred_name,
            preferred_available=bool(selection.preferred_available),
            degraded=bool(selection.degraded),
            degradation_reasons=tuple(selection.degradation_reasons),
            conformant_cid=conformant,
            supports_raw=bool(caps.supports_raw),
            supports_dag_json=bool(caps.supports_dag_json),
            supports_car=supports_car,
            supports_pin=bool(caps.supports_pin),
            codec_preservation_guaranteed=bool(
                caps.codec_preservation_guaranteed
            ),
            notes=tuple(caps.notes),
            candidate_order=tuple(selection.candidate_order),
        )


@dataclass(frozen=True)
class VerifiedPutReceipt:
    """Proof that bytes were stored under a verified CIDv1."""

    schema: str
    cid: str
    codec: str
    digest_hex: str
    byte_length: int
    backend_name: str
    backend_role: str
    rehashed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cid": self.cid,
            "codec": self.codec,
            "digest_hex": self.digest_hex,
            "byte_length": self.byte_length,
            "backend_name": self.backend_name,
            "backend_role": self.backend_role,
            "rehashed": self.rehashed,
        }


@dataclass(frozen=True)
class VerifiedGetReceipt:
    """Proof that fetched bytes rehash to the requested CIDv1."""

    schema: str
    cid: str
    codec: str
    digest_hex: str
    byte_length: int
    backend_name: str
    backend_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cid": self.cid,
            "codec": self.codec,
            "digest_hex": self.digest_hex,
            "byte_length": self.byte_length,
            "backend_name": self.backend_name,
            "backend_role": self.backend_role,
        }


@dataclass(frozen=True)
class CoordinationCidAdmission:
    """Record that a CID was admitted into a coordination manifest context."""

    schema: str
    cid: str
    codec: str
    digest_hex: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cid": self.cid,
            "codec": self.codec,
            "digest_hex": self.digest_hex,
            "purpose": self.purpose,
        }


# ---------------------------------------------------------------------------
# Core validation helpers
# ---------------------------------------------------------------------------


def _require_codec(codec: str) -> str:
    if codec not in COORDINATION_CODECS:
        raise VerifiedIPLDError(
            f"unsupported codec for coordination IPLD: {codec!r}; "
            f"allowed={sorted(COORDINATION_CODECS)}"
        )
    return codec


def admit_cid(
    value: Any,
    *,
    codecs: Iterable[str] = ("raw", "dag-json"),
) -> str:
    """Validate and return a strict CIDv1 for coordination manifests.

    Fake, truncated, uppercase, CIDv0, wrong-codec, and non-decodable strings
    fail closed via :class:`MultiformatsIdentityError` wrapped as
    :class:`VerifiedIPLDError`.
    """

    allowed = tuple(codecs)
    if not allowed or any(c not in COORDINATION_CODECS for c in allowed):
        raise VerifiedIPLDError(
            f"codecs must be a nonempty subset of {sorted(COORDINATION_CODECS)}"
        )
    try:
        return validate_cid(value, codecs=allowed)
    except MultiformatsIdentityError as exc:
        raise VerifiedIPLDError(
            f"CID rejected for coordination admission: {exc}"
        ) from exc


def expected_cid_for_bytes(data: bytes, *, codec: str = "raw") -> str:
    """Compute the local expected CIDv1 for exact bytes."""

    _require_codec(codec)
    if type(data) is not bytes:
        raise VerifiedIPLDError("payload must be exact bytes")
    try:
        return cid_for_bytes(data, codec=codec)
    except MultiformatsIdentityError as exc:
        raise VerifiedIPLDError(str(exc)) from exc


def expected_cid_for_dag_json(obj: Any) -> Tuple[str, bytes]:
    """Return (cid, canonical_bytes) for a DAG-JSON object."""

    try:
        encoded = canonical_dag_json_bytes(obj, for_identity=True)
        cid = cid_for_dag_json(obj, for_identity=True)
    except MultiformatsIdentityError as exc:
        raise VerifiedIPLDError(str(exc)) from exc
    return cid, encoded


def verify_bytes_match_cid(
    data: bytes,
    cid: str,
    *,
    codec: str = "raw",
) -> str:
    """Rehash ``data`` and require equality with ``cid`` under ``codec``."""

    _require_codec(codec)
    if type(data) is not bytes:
        raise VerifiedIPLDError("payload must be exact bytes")
    admitted = admit_cid(cid, codecs=(codec,))
    expected = expected_cid_for_bytes(data, codec=codec)
    if admitted != expected:
        raise VerifiedIPLDError(
            f"CID/byte mismatch: claimed={admitted!r} expected={expected!r} "
            f"codec={codec!r}"
        )
    return admitted


# ---------------------------------------------------------------------------
# VerifiedIPLDBackend
# ---------------------------------------------------------------------------


class VerifiedIPLDBackend:
    """Fail-closed replication adapter over an :class:`IPFSBackend`.

    Parameters
    ----------
    backend:
        Optional concrete backend.  When omitted, the router selection path is
        used and degradation is recorded on the capability receipt.
    selection:
        Optional precomputed selection receipt (tests / injection).
    require_conformant:
        When true (default), put/get for coordination refuse cache-only and
        non-conformant roles up front rather than discovering a synthetic ID.
    """

    def __init__(
        self,
        backend: Optional[ipfs_backend_router.IPFSBackend] = None,
        *,
        selection: Optional[ipfs_backend_router.BackendSelectionReceipt] = None,
        require_conformant: bool = True,
    ) -> None:
        self._require_conformant = bool(require_conformant)
        if backend is not None and selection is not None:
            self._backend = backend
            self._selection = selection
        elif backend is not None:
            self._backend, self._selection = ipfs_backend_router.select_backend(
                backend=backend
            )
        else:
            self._backend, self._selection = (
                ipfs_backend_router.get_backend_with_receipt()
            )
        self._capability = BackendCapabilityReceipt.from_selection(
            self._selection
        )

    @property
    def backend(self) -> ipfs_backend_router.IPFSBackend:
        return self._backend

    @property
    def selection(self) -> ipfs_backend_router.BackendSelectionReceipt:
        return self._selection

    def capabilities(self) -> BackendCapabilityReceipt:
        """Return the capability receipt for the bound backend."""

        return self._capability

    def _refuse_if_nonconformant(self, *, operation: str) -> None:
        cap = self._capability
        if not self._require_conformant:
            return
        if cap.role == BackendRoleName.CACHE.value or not cap.conformant_cid:
            reasons = "; ".join(cap.degradation_reasons) or "non-conformant backend"
            raise VerifiedIPLDError(
                f"{operation} refused: backend role={cap.role!r} "
                f"name={cap.backend_name!r} is not conformant for coordination "
                f"CIDs ({reasons})"
            )

    def put_raw(
        self,
        data: bytes,
        *,
        pin: bool = True,
    ) -> VerifiedPutReceipt:
        """Store exact bytes under raw CIDv1; verify backend CID and rehash."""

        return self._put_bytes(data, codec="raw", pin=pin)

    def put_dag_json(
        self,
        obj: Any,
        *,
        pin: bool = True,
    ) -> VerifiedPutReceipt:
        """Canonicalize ``obj`` as DAG-JSON and store under dag-json CIDv1."""

        _cid, encoded = expected_cid_for_dag_json(obj)
        _ = _cid
        return self._put_bytes(encoded, codec="dag-json", pin=pin)

    def _put_bytes(
        self,
        data: bytes,
        *,
        codec: str,
        pin: bool,
    ) -> VerifiedPutReceipt:
        _require_codec(codec)
        if type(data) is not bytes:
            raise VerifiedIPLDError("payload must be exact bytes")
        self._refuse_if_nonconformant(operation="put")

        expected = expected_cid_for_bytes(data, codec=codec)

        # Prefer block_put so codec can be requested; fall back to add_bytes.
        try:
            returned = self._backend.block_put(data, codec=codec)
        except TypeError:
            returned = self._backend.add_bytes(data, pin=pin)
        except RuntimeError as exc:
            raise VerifiedIPLDError(
                f"backend block_put failed for codec={codec!r}: {exc}"
            ) from exc

        # Backend-returned identifier must be a strict CIDv1 matching expected.
        try:
            returned_cid = admit_cid(returned, codecs=(codec,))
        except VerifiedIPLDError as exc:
            raise VerifiedIPLDError(
                f"backend returned non-admissible CID {returned!r} for "
                f"codec={codec!r}: {exc}"
            ) from exc

        if returned_cid != expected:
            raise VerifiedIPLDError(
                f"backend CID mismatch: returned={returned_cid!r} "
                f"expected={expected!r} codec={codec!r} "
                f"(codec substitution or non-preserving add denied)"
            )

        if pin:
            try:
                self._backend.pin(returned_cid)
            except Exception:
                # Pin is best-effort after a verified put; storage already holds
                # the block.  Re-fetch below is the admission gate.
                pass

        # Re-fetch and rehash before admitting the CID.
        try:
            fetched = self._backend.block_get(returned_cid)
        except Exception:
            try:
                fetched = self._backend.cat(returned_cid)
            except Exception as exc:
                raise VerifiedIPLDError(
                    f"backend fetch after put failed for {returned_cid!r}: {exc}"
                ) from exc

        if type(fetched) is not bytes:
            raise VerifiedIPLDError("backend fetch must return exact bytes")
        verify_bytes_match_cid(fetched, returned_cid, codec=codec)

        digest = digest_hex_from_cid(returned_cid, codecs=(codec,))
        return VerifiedPutReceipt(
            schema=PUT_RECEIPT_SCHEMA,
            cid=returned_cid,
            codec=codec,
            digest_hex=digest,
            byte_length=len(data),
            backend_name=self._capability.backend_name,
            backend_role=self._capability.role,
            rehashed=True,
        )

    def get_raw(self, cid: str) -> Tuple[bytes, VerifiedGetReceipt]:
        """Fetch bytes and rehash under raw CIDv1."""

        return self._get_bytes(cid, codec="raw")

    def get_dag_json(self, cid: str) -> Tuple[bytes, VerifiedGetReceipt]:
        """Fetch DAG-JSON bytes and rehash under dag-json CIDv1."""

        data, receipt = self._get_bytes(cid, codec="dag-json")
        # Ensure the bytes are canonical DAG-JSON.
        try:
            require_canonical_dag_json_bytes(data)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(
                f"fetched dag-json bytes are not canonical: {exc}"
            ) from exc
        return data, receipt

    def _get_bytes(
        self,
        cid: str,
        *,
        codec: str,
    ) -> Tuple[bytes, VerifiedGetReceipt]:
        _require_codec(codec)
        self._refuse_if_nonconformant(operation="get")
        admitted = admit_cid(cid, codecs=(codec,))

        try:
            data = self._backend.block_get(admitted)
        except Exception:
            try:
                data = self._backend.cat(admitted)
            except Exception as exc:
                raise VerifiedIPLDError(
                    f"backend fetch failed for {admitted!r}: {exc}"
                ) from exc

        if type(data) is not bytes:
            raise VerifiedIPLDError("backend fetch must return exact bytes")
        verify_bytes_match_cid(data, admitted, codec=codec)
        digest = digest_hex_from_cid(admitted, codecs=(codec,))
        receipt = VerifiedGetReceipt(
            schema=GET_RECEIPT_SCHEMA,
            cid=admitted,
            codec=codec,
            digest_hex=digest,
            byte_length=len(data),
            backend_name=self._capability.backend_name,
            backend_role=self._capability.role,
        )
        return data, receipt

    def admit_for_manifest(
        self,
        cid: str,
        *,
        codec: str,
        payload: Optional[bytes] = None,
        purpose: str = "coordination-manifest",
    ) -> CoordinationCidAdmission:
        """Admit a CID into a coordination manifest after strict checks.

        When ``payload`` is supplied it is rehashed against the CID.  Unsupported
        codecs and non-CIDv1 strings fail closed.
        """

        _require_codec(codec)
        admitted = admit_cid(cid, codecs=(codec,))
        if payload is not None:
            verify_bytes_match_cid(payload, admitted, codec=codec)
        digest = digest_hex_from_cid(admitted, codecs=(codec,))
        return CoordinationCidAdmission(
            schema=ADMISSION_RECEIPT_SCHEMA,
            cid=admitted,
            codec=codec,
            digest_hex=digest,
            purpose=str(purpose),
        )

    def export_car(self, cid: str, *, codec: str = "raw") -> bytes:
        """Capability-gated CAR export; fails closed when unsupported."""

        cap = self._capability
        if not cap.supports_car:
            raise VerifiedIPLDError(
                f"CAR export unsupported for backend role={cap.role!r} "
                f"name={cap.backend_name!r}; degradation="
                f"{list(cap.degradation_reasons)}"
            )
        admitted = admit_cid(cid, codecs=(codec,) if codec else ("raw", "dag-json"))
        # Optional: ensure we still hold bytes that rehash (when fetchable).
        try:
            data = self._backend.block_get(admitted)
        except Exception:
            try:
                data = self._backend.cat(admitted)
            except Exception as exc:
                raise VerifiedIPLDError(
                    f"CAR export precondition fetch failed for {admitted!r}: {exc}"
                ) from exc
        if type(data) is bytes and codec in COORDINATION_CODECS:
            verify_bytes_match_cid(data, admitted, codec=codec)

        try:
            car = self._backend.dag_export(admitted)
        except Exception as exc:
            raise VerifiedIPLDError(
                f"CAR export failed (capability claimed but backend error): {exc}"
            ) from exc
        if type(car) is not bytes or not car:
            raise VerifiedIPLDError("CAR export returned empty or non-bytes payload")
        return car

    # -- Identity bridges (runtime-CAS / MCP++ hashes) ---------------------

    def link_runtime_cas(
        self,
        artifact_id: str,
        *,
        payload_bytes: Optional[bytes] = None,
        payload_digest: Optional[str] = None,
        codec: str = "raw",
    ) -> IdentityLink:
        """Bridge a runtime-CAS artifact id to a CIDv1 without replacing it."""

        try:
            return link_runtime_artifact(
                artifact_id,
                payload_bytes=payload_bytes,
                payload_digest=payload_digest,
                codec=codec,
            )
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def link_mcp_payload_digest(
        self,
        payload_digest: str,
        *,
        codec: str = "raw",
    ) -> IdentityLink:
        """Bridge an MCP++ / compaction ``sha256:…`` digest via IdentityLink."""

        try:
            return link_payload_digest(payload_digest, codec=codec)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def link_raw_payload(
        self,
        data: bytes,
        *,
        local_id: Optional[str] = None,
    ) -> IdentityLink:
        """Address exact bytes and optionally retain a local label."""

        try:
            return link_raw_bytes(data, local_id=local_id)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc


def open_verified_ipld_backend(
    *,
    backend: Optional[ipfs_backend_router.IPFSBackend] = None,
    require_conformant: bool = True,
) -> VerifiedIPLDBackend:
    """Factory for the default verified replication adapter."""

    return VerifiedIPLDBackend(
        backend=backend,
        require_conformant=require_conformant,
    )


__all__ = [
    "ADMISSION_RECEIPT_SCHEMA",
    "ALLOWED_CODECS",
    "BACKEND_CAPABILITY_RECEIPT_SCHEMA",
    "BackendCapabilityReceipt",
    "BackendRoleName",
    "CID_BASE",
    "CID_VERSION",
    "COORDINATION_CODECS",
    "DIGEST_SIZE",
    "GET_RECEIPT_SCHEMA",
    "IDENTITY_LINK_SCHEMA",
    "IdentityKind",
    "IdentityLink",
    "InMemoryConformantBackend",
    "MH_TYPE",
    "PUT_RECEIPT_SCHEMA",
    "VERIFIED_IPLD_SCHEMA",
    "VerifiedGetReceipt",
    "VerifiedIPLDBackend",
    "VerifiedIPLDError",
    "VerifiedPutReceipt",
    "CoordinationCidAdmission",
    "admit_cid",
    "expected_cid_for_bytes",
    "expected_cid_for_dag_json",
    "open_verified_ipld_backend",
    "verify_bytes_match_cid",
]
