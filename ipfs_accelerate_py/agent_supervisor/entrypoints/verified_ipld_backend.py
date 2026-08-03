"""Verified CIDv1 / IPLD / IPFS replication adapter (ASE-038).

Coordination manifests may only reference strict CIDv1 objects under the frozen
multiformats profile (CIDv1 / lowercase base32 / raw|dag-json / sha2-256).
This adapter:

* computes expected raw or DAG-JSON CIDs locally before trusting any transport;
* validates backend-returned identifiers and rehashes fetched bytes;
* capability-gates CAR export;
* classifies ``ipfs_kit_py``, Kubo, and cache roles accurately with explicit
  degradation (Hugging Face remains cache-only until strict-CID conformant);
* bridges runtime-CAS and MCP++ digest identities through explicit
  :class:`~ipfs_accelerate_py.agent_supervisor.multiformats_identity.IdentityLink`
  records without treating those digests as coordination authority.

Fake, truncated, or mismatched CIDs and unsupported codecs/CAR fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Iterable, Optional

from ipfs_accelerate_py import ipfs_backend_router
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
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
    link_runtime_artifact,
    parse_payload_digest,
    validate_cid,
)

VERIFIED_IPLD_BACKEND_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/verified-ipld-backend@1"
)
BACKEND_CAPABILITY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/backend-capability-receipt@1"
)
VERIFIED_PUT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/verified-put-receipt@1"
)
VERIFIED_IPLD_REQUIREMENT_ID: Final = (
    "coordination_replication.VERIFIED_IPLD_BACKEND_REQUIREMENT_ID"
)

# Closed codec vocabulary for coordination replication.  Anything else fails.
COORDINATION_CODECS: Final = frozenset({"raw", "dag-json"})
MANIFEST_ADMITTED_CODECS: Final = COORDINATION_CODECS


class VerifiedIPLDError(ValueError):
    """A verified IPLD / CID / CAR operation failed closed."""


class BackendRoleName(str, Enum):
    """Mirror of router roles for receipt serialization."""

    IPFS_KIT = "ipfs_kit_py"
    KUBO = "kubo"
    CACHE = "cache"
    UNKNOWN = "unknown"


def _role_name(role: Any) -> BackendRoleName:
    value = getattr(role, "value", role)
    text = str(value or BackendRoleName.UNKNOWN.value)
    try:
        return BackendRoleName(text)
    except ValueError:
        return BackendRoleName.UNKNOWN


@dataclass(frozen=True)
class BackendCapabilityReceipt:
    """Capability matrix receipt for the active replication transport.

    Cache roles always set ``admits_coordination_manifest`` false and record
    degradation.  CAR support is capability-gated, never assumed from role.
    """

    SCHEMA: ClassVar[str] = BACKEND_CAPABILITY_RECEIPT_SCHEMA

    backend_name: str
    role: str
    synthetic_identifiers: bool
    admits_coordination_manifest: bool
    supports_raw: bool
    supports_dag_json: bool
    supports_car: bool
    preserves_requested_codec: bool
    pin_supported: bool
    degraded: bool
    degradation_reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    selection_degraded: bool = False
    selection_reasons: tuple[str, ...] = ()
    schema: str = BACKEND_CAPABILITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BACKEND_CAPABILITY_RECEIPT_SCHEMA:
            raise VerifiedIPLDError("unsupported backend capability receipt schema")
        if not isinstance(self.backend_name, str) or not self.backend_name:
            raise VerifiedIPLDError("backend_name must be a nonempty string")
        try:
            BackendRoleName(self.role)
        except ValueError as exc:
            raise VerifiedIPLDError(f"unsupported backend role: {self.role!r}") from exc
        if self.role == BackendRoleName.CACHE.value:
            if self.admits_coordination_manifest:
                raise VerifiedIPLDError(
                    "cache role cannot admit coordination-manifest CIDs"
                )
            if not self.synthetic_identifiers and not self.degraded:
                raise VerifiedIPLDError(
                    "cache role must report synthetic identifiers or degradation"
                )
        if self.admits_coordination_manifest and self.synthetic_identifiers:
            raise VerifiedIPLDError(
                "synthetic identifier transports cannot admit coordination CIDs"
            )
        for flag in (
            "synthetic_identifiers",
            "admits_coordination_manifest",
            "supports_raw",
            "supports_dag_json",
            "supports_car",
            "preserves_requested_codec",
            "pin_supported",
            "degraded",
            "selection_degraded",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise VerifiedIPLDError(f"{flag} must be a boolean")
        object.__setattr__(
            self,
            "degradation_reasons",
            tuple(str(item) for item in self.degradation_reasons if str(item)),
        )
        object.__setattr__(
            self,
            "selection_reasons",
            tuple(str(item) for item in self.selection_reasons if str(item)),
        )
        object.__setattr__(
            self,
            "notes",
            tuple(str(item) for item in self.notes if str(item)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend_name": self.backend_name,
            "role": self.role,
            "synthetic_identifiers": self.synthetic_identifiers,
            "admits_coordination_manifest": self.admits_coordination_manifest,
            "supports_raw": self.supports_raw,
            "supports_dag_json": self.supports_dag_json,
            "supports_car": self.supports_car,
            "preserves_requested_codec": self.preserves_requested_codec,
            "pin_supported": self.pin_supported,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "notes": list(self.notes),
            "selection_degraded": self.selection_degraded,
            "selection_reasons": list(self.selection_reasons),
            "content_id": self.content_id,
        }

    @property
    def content_id(self) -> str:
        payload = {
            "schema": self.schema,
            "backend_name": self.backend_name,
            "role": self.role,
            "synthetic_identifiers": self.synthetic_identifiers,
            "admits_coordination_manifest": self.admits_coordination_manifest,
            "supports_raw": self.supports_raw,
            "supports_dag_json": self.supports_dag_json,
            "supports_car": self.supports_car,
            "preserves_requested_codec": self.preserves_requested_codec,
            "pin_supported": self.pin_supported,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "notes": list(self.notes),
            "selection_degraded": self.selection_degraded,
            "selection_reasons": list(self.selection_reasons),
        }
        return cid_for_dag_json(payload, for_identity=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendCapabilityReceipt":
        if not isinstance(value, Mapping):
            raise VerifiedIPLDError("capability receipt must be an object")
        allowed = {
            "schema",
            "backend_name",
            "role",
            "synthetic_identifiers",
            "admits_coordination_manifest",
            "supports_raw",
            "supports_dag_json",
            "supports_car",
            "preserves_requested_codec",
            "pin_supported",
            "degraded",
            "degradation_reasons",
            "notes",
            "selection_degraded",
            "selection_reasons",
            "content_id",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise VerifiedIPLDError(
                f"capability receipt contains unknown fields: {sorted(unknown)}"
            )
        receipt = cls(
            schema=str(value.get("schema") or BACKEND_CAPABILITY_RECEIPT_SCHEMA),
            backend_name=str(value.get("backend_name") or ""),
            role=str(value.get("role") or ""),
            synthetic_identifiers=bool(value.get("synthetic_identifiers")),
            admits_coordination_manifest=bool(
                value.get("admits_coordination_manifest")
            ),
            supports_raw=bool(value.get("supports_raw")),
            supports_dag_json=bool(value.get("supports_dag_json")),
            supports_car=bool(value.get("supports_car")),
            preserves_requested_codec=bool(value.get("preserves_requested_codec")),
            pin_supported=bool(value.get("pin_supported")),
            degraded=bool(value.get("degraded")),
            degradation_reasons=tuple(value.get("degradation_reasons") or ()),
            notes=tuple(value.get("notes") or ()),
            selection_degraded=bool(value.get("selection_degraded")),
            selection_reasons=tuple(value.get("selection_reasons") or ()),
        )
        claimed = value.get("content_id")
        if claimed is not None and str(claimed) and str(claimed) != receipt.content_id:
            raise VerifiedIPLDError(
                "capability receipt content_id does not match payload"
            )
        return receipt


@dataclass(frozen=True)
class VerifiedPutReceipt:
    """Evidence that bytes were stored and re-fetched under a strict CIDv1."""

    cid: str
    codec: str
    byte_length: int
    digest_hex: str
    transport_handle: str
    transport_matched_cid: bool
    backend_name: str
    backend_role: str
    pin: bool
    schema: str = VERIFIED_PUT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedIPLDBackend:
    """Fail-closed replication adapter over an :class:`IPFSBackend` transport.

    The transport may be ipfs_kit_py, Kubo, or a local cache.  Only verified
    strict CIDv1 strings produced or validated here may enter coordination
    manifests.
    """

    transport: Any = None
    selection_receipt: Optional[ipfs_backend_router.BackendSelectionReceipt] = None
    _transport_handles: dict[str, str] = field(default_factory=dict, repr=False)
    _local_blocks: dict[str, bytes] = field(default_factory=dict, repr=False)
    _local_codecs: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.transport is None:
            chosen, receipt = ipfs_backend_router.select_backend_with_receipt()
            self.transport = chosen
            self.selection_receipt = receipt
        elif self.selection_receipt is None:
            _chosen, receipt = ipfs_backend_router.select_backend_with_receipt(
                backend=self.transport
            )
            self.selection_receipt = receipt

    # ------------------------------------------------------------------
    # Capability / role reporting
    # ------------------------------------------------------------------

    def capability_receipt(self) -> BackendCapabilityReceipt:
        """Return an accurate role/capability receipt with explicit degradation."""
        descriptor = ipfs_backend_router.describe_backend(self.transport)
        role = _role_name(descriptor.role)
        selection = self.selection_receipt
        selection_degraded = bool(selection.degraded) if selection else False
        selection_reasons = (
            tuple(selection.degradation_reasons) if selection else ()
        )
        # Cache and synthetic transports never admit coordination manifests.
        admits = (
            bool(descriptor.admits_strict_cid)
            and not descriptor.synthetic_identifiers
            and role is not BackendRoleName.CACHE
        )
        degraded = bool(descriptor.degraded) or selection_degraded or not admits
        reasons: list[str] = list(descriptor.degradation_reasons)
        reasons.extend(selection_reasons)
        if role is BackendRoleName.CACHE:
            reasons.append("cache_role_not_coordination_authority")
        if descriptor.synthetic_identifiers:
            reasons.append("synthetic_identifiers_not_strict_cid")
        if not descriptor.supports_car:
            reasons.append("car_export_unsupported")
        if not descriptor.preserves_requested_codec:
            reasons.append("codec_preservation_not_assumed")
        # Stable unique order
        unique_reasons = tuple(dict.fromkeys(r for r in reasons if r))
        return BackendCapabilityReceipt(
            backend_name=descriptor.name,
            role=role.value,
            synthetic_identifiers=bool(descriptor.synthetic_identifiers),
            admits_coordination_manifest=admits,
            supports_raw=bool(descriptor.supports_raw),
            supports_dag_json=bool(descriptor.supports_dag_json),
            supports_car=bool(descriptor.supports_car),
            preserves_requested_codec=bool(descriptor.preserves_requested_codec),
            pin_supported=bool(descriptor.pin_supported),
            degraded=degraded,
            degradation_reasons=unique_reasons,
            notes=tuple(descriptor.notes),
            selection_degraded=selection_degraded,
            selection_reasons=selection_reasons,
        )

    def backend_role(self) -> str:
        """Return the closed role name for the active transport."""
        return self.capability_receipt().role

    # ------------------------------------------------------------------
    # CID admission (manifest gate)
    # ------------------------------------------------------------------

    def admit_cid_for_manifest(
        self,
        value: Any,
        *,
        codecs: Iterable[str] = ("raw", "dag-json"),
    ) -> str:
        """Admit only a strict CIDv1 into a coordination manifest.

        Fake, truncated, mismatched-case, or wrong-codec identifiers fail
        closed.  Cache synthetic keys never pass validation.
        """
        allowed = tuple(codecs) if codecs is not None else ("raw", "dag-json")
        if not allowed:
            raise VerifiedIPLDError("manifest admission requires at least one codec")
        for codec in allowed:
            if codec not in MANIFEST_ADMITTED_CODECS:
                raise VerifiedIPLDError(
                    f"unsupported coordination codec for manifest admission: {codec!r}"
                )
        try:
            return validate_cid(value, codecs=allowed)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(
                f"CID rejected for coordination manifest: {exc}"
            ) from exc

    def admit_cid_list_for_manifest(
        self,
        values: Sequence[Any],
        *,
        codecs: Iterable[str] = ("raw", "dag-json"),
    ) -> tuple[str, ...]:
        """Admit an ordered list of strict CIDv1 objects for a manifest."""
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise VerifiedIPLDError("manifest CID list must be a sequence of CIDs")
        admitted: list[str] = []
        seen: set[str] = set()
        for item in values:
            cid = self.admit_cid_for_manifest(item, codecs=codecs)
            if cid in seen:
                raise VerifiedIPLDError(
                    f"duplicate CID in coordination manifest: {cid}"
                )
            seen.add(cid)
            admitted.append(cid)
        return tuple(admitted)

    # ------------------------------------------------------------------
    # Local identity + put/get with rehash
    # ------------------------------------------------------------------

    def expected_cid_for_bytes(self, data: bytes, *, codec: str = "raw") -> str:
        """Compute the local expected CIDv1 for exact bytes and codec."""
        codec = self._require_codec(codec)
        if type(data) is not bytes:
            raise VerifiedIPLDError("payload must be exact bytes")
        try:
            return cid_for_bytes(data, codec=codec)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def expected_cid_for_dag_json(
        self,
        obj: Any,
        *,
        for_identity: bool = True,
    ) -> str:
        """Compute the local expected CIDv1 for a DAG-JSON object."""
        try:
            return cid_for_dag_json(obj, for_identity=for_identity)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def put_raw(
        self,
        data: bytes,
        *,
        pin: bool = True,
    ) -> VerifiedPutReceipt:
        """Store exact bytes, require rehash match, return verified raw CIDv1."""
        return self._put_bytes(data, codec="raw", pin=pin)

    def put_dag_json(
        self,
        obj: Any,
        *,
        pin: bool = True,
        for_identity: bool = True,
    ) -> VerifiedPutReceipt:
        """Store canonical DAG-JSON bytes under a verified dag-json CIDv1."""
        try:
            encoded = canonical_dag_json_bytes(obj, for_identity=for_identity)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc
        return self._put_bytes(encoded, codec="dag-json", pin=pin)

    def _put_bytes(
        self,
        data: bytes,
        *,
        codec: str,
        pin: bool,
    ) -> VerifiedPutReceipt:
        codec = self._require_codec(codec)
        if type(data) is not bytes:
            raise VerifiedIPLDError("payload must be exact bytes")
        expected = self.expected_cid_for_bytes(data, codec=codec)
        capability = self.capability_receipt()

        # Unsupported codec relative to transport capability fails closed when
        # the transport claims codec preservation for something else only.
        if codec == "raw" and not capability.supports_raw:
            raise VerifiedIPLDError("transport does not support raw blocks")
        if codec == "dag-json" and not (
            capability.supports_dag_json or capability.supports_raw
        ):
            # DAG-JSON manifests may be stored as exact encoded bytes on raw-
            # capable transports; pure non-byte stores fail closed.
            raise VerifiedIPLDError("transport cannot store dag-json payloads")

        transport_handle = self._transport_put(data, codec=codec, pin=pin)
        fetched = self._transport_get(transport_handle)
        if fetched != data:
            raise VerifiedIPLDError(
                "fetched bytes do not match stored payload (transport corruption)"
            )
        recomputed = self.expected_cid_for_bytes(fetched, codec=codec)
        if recomputed != expected:
            raise VerifiedIPLDError(
                "local rehash after put diverged from expected CID"
            )

        transport_matched = False
        if transport_handle == expected:
            transport_matched = True
        else:
            # Attempt to treat the transport handle as a CID under the same
            # codec; codec substitution or synthetic keys fail closed when the
            # transport claims strict CID admission.
            try:
                validated_handle = validate_cid(transport_handle, codecs=(codec,))
            except MultiformatsIdentityError:
                validated_handle = None
            if validated_handle is not None and validated_handle != expected:
                # Same multihash / wrong codec or wrong digest.
                raise VerifiedIPLDError(
                    "backend returned a mismatched CIDv1 for the stored bytes "
                    f"(expected {expected}, got {validated_handle})"
                )
            if capability.admits_coordination_manifest and not capability.synthetic_identifiers:
                # Conformant transports must return the exact expected CID.
                raise VerifiedIPLDError(
                    "conformant backend did not return the expected strict CIDv1 "
                    f"(expected {expected}, got {transport_handle!r})"
                )
            # Cache / non-conformant path: keep a handle map; never admit the
            # synthetic key as the coordination CID.
            transport_matched = False

        self._transport_handles[expected] = transport_handle
        self._local_blocks[expected] = data
        self._local_codecs[expected] = codec

        if pin and capability.pin_supported:
            try:
                self.transport.pin(transport_handle)
            except Exception:
                # Pin failure is non-fatal for local verification; identity is
                # established by rehash, not pin state.
                pass

        return VerifiedPutReceipt(
            cid=expected,
            codec=codec,
            byte_length=len(data),
            digest_hex=digest_hex_from_cid(expected, codecs=(codec,)),
            transport_handle=transport_handle,
            transport_matched_cid=transport_matched,
            backend_name=capability.backend_name,
            backend_role=capability.role,
            pin=bool(pin),
        )

    def get_verified(
        self,
        cid: str,
        *,
        codec: str = "raw",
    ) -> bytes:
        """Fetch bytes for a strict CIDv1 and rehash before returning them."""
        codec = self._require_codec(codec)
        try:
            canonical = validate_cid(cid, codecs=(codec,))
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(f"CID rejected before fetch: {exc}") from exc

        if canonical in self._local_blocks:
            data = self._local_blocks[canonical]
        else:
            handle = self._transport_handles.get(canonical, canonical)
            data = self._transport_get(handle)

        if type(data) is not bytes:
            raise VerifiedIPLDError("transport returned non-bytes payload")
        recomputed = self.expected_cid_for_bytes(data, codec=codec)
        if recomputed != canonical:
            raise VerifiedIPLDError(
                "fetched bytes rehash does not match requested CID "
                f"(expected {canonical}, got {recomputed})"
            )
        self._local_blocks[canonical] = data
        self._local_codecs[canonical] = codec
        return data

    def get_dag_json(self, cid: str) -> Any:
        """Fetch and parse a verified dag-json block as a JSON value."""
        import json

        raw = self.get_verified(cid, codec="dag-json")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedIPLDError(
                "verified dag-json block is not valid UTF-8 JSON"
            ) from exc

    # ------------------------------------------------------------------
    # CAR capability gate
    # ------------------------------------------------------------------

    def export_car(self, cid: str, *, codec: str = "dag-json") -> bytes:
        """Export a CAR only when the transport capability matrix allows it."""
        capability = self.capability_receipt()
        if not capability.supports_car:
            raise VerifiedIPLDError(
                "CAR export is capability-gated and unsupported by the active "
                f"backend role={capability.role!r} name={capability.backend_name!r}"
            )
        # Validate root CID form first; unsupported codecs fail closed.
        admitted = self.admit_cid_for_manifest(cid, codecs=(codec, "raw", "dag-json"))
        try:
            car_bytes = self.transport.dag_export(admitted)
        except Exception as exc:
            raise VerifiedIPLDError(
                f"CAR export failed closed: {type(exc).__name__}: {exc}"
            ) from exc
        if type(car_bytes) is not bytes or not car_bytes:
            raise VerifiedIPLDError("CAR export returned empty or non-bytes payload")
        return car_bytes

    # ------------------------------------------------------------------
    # Identity links (runtime-CAS / MCP++ digests)
    # ------------------------------------------------------------------

    def link_runtime_cas(
        self,
        artifact_id: str,
        *,
        payload_bytes: bytes | None = None,
        payload_digest: str | None = None,
        codec: str = "raw",
    ) -> IdentityLink:
        """Bridge a runtime-CAS artifact id to a strict CIDv1 via IdentityLink."""
        try:
            return link_runtime_artifact(
                artifact_id,
                payload_bytes=payload_bytes,
                payload_digest=payload_digest,
                codec=codec,
            )
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def link_mcp_compaction_hash(
        self,
        digest: str,
        *,
        codec: str = "raw",
    ) -> IdentityLink:
        """Bridge an MCP++ ``sha256:…`` compaction hash to a strict CIDv1.

        The link is explicit dual identity; the digest string is never treated
        as coordination-epoch authority by itself.
        """
        text = digest if isinstance(digest, str) else ""
        if text and not text.startswith("sha256:"):
            # Accept bare 64-hex and normalize to the payload-digest form.
            try:
                parse_payload_digest(f"sha256:{text.lower()}")
            except MultiformatsIdentityError as exc:
                raise VerifiedIPLDError(
                    "MCP++ compaction hash must be sha256:<64-hex> or 64-hex"
                ) from exc
            text = f"sha256:{text.lower()}"
        try:
            return link_payload_digest(text, codec=codec)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    def link_raw_identity(
        self,
        data: bytes,
        *,
        local_id: str | None = None,
    ) -> IdentityLink:
        """Create a raw-bytes IdentityLink under the frozen profile."""
        from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
            link_raw_bytes,
        )

        try:
            return link_raw_bytes(data, local_id=local_id)
        except MultiformatsIdentityError as exc:
            raise VerifiedIPLDError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _require_codec(self, codec: str) -> str:
        if codec not in COORDINATION_CODECS:
            raise VerifiedIPLDError(
                f"unsupported codec {codec!r}; coordination allows only "
                f"{sorted(COORDINATION_CODECS)}"
            )
        if codec not in ALLOWED_CODECS:
            raise VerifiedIPLDError(f"codec {codec!r} is outside the multiformats profile")
        return codec

    def _transport_put(self, data: bytes, *, codec: str, pin: bool) -> str:
        """Store bytes on the transport with the declared codec when possible.

        Prefer ``block_put(..., codec=…)`` so conformant transports can return
        a CIDv1 under the requested codec.  Fall back to ``add_bytes`` only
        when block APIs are absent; callers still rehash and reject codec
        substitution on conformant roles.
        """
        try:
            if hasattr(self.transport, "block_put"):
                return str(self.transport.block_put(data, codec=codec))
            return str(self.transport.add_bytes(data, pin=pin))
        except VerifiedIPLDError:
            raise
        except Exception as exc:
            raise VerifiedIPLDError(
                f"transport put failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _transport_get(self, handle: str) -> bytes:
        try:
            if hasattr(self.transport, "block_get"):
                try:
                    return self.transport.block_get(handle)
                except Exception:
                    pass
            return self.transport.cat(handle)
        except VerifiedIPLDError:
            raise
        except Exception as exc:
            raise VerifiedIPLDError(
                f"transport get failed: {type(exc).__name__}: {exc}"
            ) from exc


def build_verified_ipld_backend(
    *,
    transport: Any | None = None,
    deps: object = None,
) -> VerifiedIPLDBackend:
    """Construct a verified adapter, optionally selecting via the router."""
    if transport is not None:
        return VerifiedIPLDBackend(transport=transport)
    chosen, receipt = ipfs_backend_router.select_backend_with_receipt(deps=deps)
    return VerifiedIPLDBackend(transport=chosen, selection_receipt=receipt)


__all__ = (
    "ALLOWED_CODECS",
    "BACKEND_CAPABILITY_RECEIPT_SCHEMA",
    "BackendCapabilityReceipt",
    "BackendRoleName",
    "CID_BASE",
    "CID_VERSION",
    "COORDINATION_CODECS",
    "DIGEST_SIZE",
    "IDENTITY_LINK_SCHEMA",
    "IdentityKind",
    "IdentityLink",
    "MANIFEST_ADMITTED_CODECS",
    "MH_TYPE",
    "VERIFIED_IPLD_BACKEND_SCHEMA",
    "VERIFIED_IPLD_REQUIREMENT_ID",
    "VERIFIED_PUT_RECEIPT_SCHEMA",
    "VerifiedIPLDBackend",
    "VerifiedIPLDError",
    "VerifiedPutReceipt",
    "build_verified_ipld_backend",
)
