"""Libp2pDiscovery@1 — Registry@1 adapter over a libp2p-style discovery mesh.

Provides publish / refresh / withdraw / lookup / health-aware selection for
AgentAdvertisement@1 records. A local store holds validated records; a
transport mesh gossips lifecycle events so peers can discover each other.

Hermetic tests inject :class:`InMemoryLibp2pMesh` — no real network or
py-libp2p host is required. Production callers may supply any object that
implements the :class:`Libp2pDiscoveryTransport` protocol (broadcast /
subscribe). Registry membership is never execution authority.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    REGISTRY_INTERFACE,
    Registry,
    RegistryError,
    RegistryNotFoundError,
    RegistryStaleError,
    RegistryValidationError,
    identity_did,
    is_stale,
    policy_languages_of,
    proof_systems_of,
    select_advertisement,
    semantic_capabilities_of,
    sort_for_selection,
    validate_agent_advertisement,
)

LIBP2P_DISCOVERY_INTERFACE = "Libp2pDiscovery@1"
LIBP2P_PROVIDER_ID = "libp2p"

# Versioned stream / topic protocol id for advertisement gossip (Profile E style).
LIBP2P_DISCOVERY_PROTOCOL_ID = "/mcp++/discovery/agent-advertisement/1.0.0"

# Lifecycle ops carried on the mesh.
OP_PUBLISH = "publish"
OP_REFRESH = "refresh"
OP_WITHDRAW = "withdraw"
_KNOWN_OPS = frozenset({OP_PUBLISH, OP_REFRESH, OP_WITHDRAW})


class Libp2pDiscoveryError(RegistryError):
    """Base error for Libp2pDiscovery@1 transport / mesh failures."""


class Libp2pTransportError(Libp2pDiscoveryError):
    """Raised when the discovery transport cannot deliver a message."""


@runtime_checkable
class Libp2pDiscoveryTransport(Protocol):
    """Minimal transport surface used by :class:`Libp2pDiscovery`.

    Implementations MUST be safe for concurrent ``broadcast`` and handler
    callbacks. Handlers receive the raw mesh message mapping.
    """

    def broadcast(self, message: Mapping[str, Any]) -> None:
        """Deliver ``message`` to all subscribed peers (including optional local)."""

    def subscribe(self, peer_id: str, handler: Callable[[Mapping[str, Any]], None]) -> None:
        """Register ``handler`` for messages delivered to ``peer_id``."""

    def unsubscribe(self, peer_id: str) -> None:
        """Remove the handler for ``peer_id`` if present."""


def _wall_clock_ms() -> int:
    return int(time.time() * 1000)


def _require_non_empty_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{field} must be a non-empty string")
    return value.strip()


class InMemoryLibp2pMesh:
    """Deterministic in-process multi-peer mesh for hermetic discovery tests.

    Peers share one mesh instance. ``broadcast`` delivers a deep copy of the
    message to every subscribed peer except the origin (when ``origin_peer_id``
    is set on the message). Delivery is synchronous so unit tests need no
    sleeps or event loops.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {}
        self._broadcast_count = 0
        self._deliver_count = 0

    def subscribe(self, peer_id: str, handler: Callable[[Mapping[str, Any]], None]) -> None:
        peer = _require_non_empty_str(peer_id, field="peer_id")
        if not callable(handler):
            raise Libp2pTransportError("handler must be callable")
        with self._lock:
            self._handlers[peer] = handler

    def unsubscribe(self, peer_id: str) -> None:
        peer = _require_non_empty_str(peer_id, field="peer_id")
        with self._lock:
            self._handlers.pop(peer, None)

    def broadcast(self, message: Mapping[str, Any]) -> None:
        if not isinstance(message, Mapping):
            raise Libp2pTransportError("message must be a mapping")
        origin = message.get("origin_peer_id")
        with self._lock:
            self._broadcast_count += 1
            targets = list(self._handlers.items())
        for peer_id, handler in targets:
            if origin is not None and peer_id == origin:
                continue
            try:
                handler(copy.deepcopy(dict(message)))
                with self._lock:
                    self._deliver_count += 1
            except Exception:
                # One bad peer must not block the mesh; adapter logs via stats.
                continue

    def peer_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._handlers.keys())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "peers": len(self._handlers),
                "broadcast_count": self._broadcast_count,
                "deliver_count": self._deliver_count,
            }


class Libp2pDiscovery(Registry):
    """Registry@1 adapter that gossips advertisements over a libp2p mesh.

    Parameters
    ----------
    transport:
        Mesh implementing :class:`Libp2pDiscoveryTransport`. Defaults to a
        private :class:`InMemoryLibp2pMesh` (single-peer, hermetic).
    peer_id:
        Stable peer label on the mesh. Auto-generated when omitted.
    require_signed:
        When True, reject advertisements lacking a signature block shape.
    clock_ms:
        Injectable epoch-millisecond clock for hermetic TTL tests.
    accept_remote:
        When False, ignore inbound mesh messages (local-only mode).
    """

    def __init__(
        self,
        *,
        transport: Optional[Libp2pDiscoveryTransport] = None,
        peer_id: Optional[str] = None,
        require_signed: bool = False,
        clock_ms: Optional[Callable[[], int]] = None,
        accept_remote: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._receipt_ms: dict[str, int] = {}
        self._require_signed = bool(require_signed)
        self._clock_ms = clock_ms if clock_ms is not None else _wall_clock_ms
        self._accept_remote = bool(accept_remote)
        self._transport: Libp2pDiscoveryTransport = (
            transport if transport is not None else InMemoryLibp2pMesh()
        )
        self._peer_id = (
            peer_id.strip()
            if isinstance(peer_id, str) and peer_id.strip()
            else f"peer-{uuid.uuid4().hex[:12]}"
        )
        self._publish_count = 0
        self._refresh_count = 0
        self._withdraw_count = 0
        self._stale_rejects = 0
        self._unsigned_rejects = 0
        self._remote_ingests = 0
        self._remote_rejects = 0
        self._broadcasts = 0
        self._closed = False

        self._transport.subscribe(self._peer_id, self._on_remote_message)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return LIBP2P_PROVIDER_ID

    @property
    def interface(self) -> str:
        return LIBP2P_DISCOVERY_INTERFACE

    @property
    def family_interface(self) -> str:
        """Parent Registry@1 family label."""

        return REGISTRY_INTERFACE

    @property
    def peer_id(self) -> str:
        return self._peer_id

    @property
    def protocol_id(self) -> str:
        return LIBP2P_DISCOVERY_PROTOCOL_ID

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Unsubscribe from the mesh. Idempotent."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._transport.unsubscribe(self._peer_id)
        except Exception:
            pass

    def __enter__(self) -> "Libp2pDiscovery":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Clock / normalize / store (mirrors StaticRegistry)
    # ------------------------------------------------------------------

    def _now(self, now_ms: Optional[int]) -> int:
        if now_ms is not None:
            if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
                raise RegistryValidationError("now_ms must be a non-negative integer")
            return now_ms
        value = self._clock_ms()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RegistryValidationError("clock_ms() must return a non-negative integer")
        return value

    def _normalize(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> dict[str, Any]:
        try:
            return validate_agent_advertisement(
                advertisement,
                require_signed=self._require_signed,
                now_ms=now_ms,
                reject_stale=True,
                receipt_ms=now_ms,
            )
        except RegistryStaleError:
            self._stale_rejects += 1
            raise
        except Exception as exc:
            from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
                RegistryUnsignedError,
            )

            if isinstance(exc, RegistryUnsignedError):
                self._unsigned_rejects += 1
            raise

    def _store(self, normalized: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
        did = identity_did(normalized)
        stored = copy.deepcopy(normalized)
        if "published_at_ms" not in stored:
            stored["published_at_ms"] = now_ms
        self._records[did] = stored
        self._receipt_ms[did] = now_ms
        return copy.deepcopy(stored)

    def _broadcast(self, op: str, *, advertisement: Optional[Mapping[str, Any]] = None,
                   identity: Optional[str] = None, now_ms: int) -> None:
        message: dict[str, Any] = {
            "protocol_id": LIBP2P_DISCOVERY_PROTOCOL_ID,
            "op": op,
            "origin_peer_id": self._peer_id,
            "now_ms": now_ms,
        }
        if advertisement is not None:
            message["advertisement"] = copy.deepcopy(dict(advertisement))
        if identity is not None:
            message["identity_did"] = identity
        try:
            self._transport.broadcast(message)
            self._broadcasts += 1
        except Libp2pTransportError:
            raise
        except Exception as exc:
            raise Libp2pTransportError(f"mesh broadcast failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Remote ingest
    # ------------------------------------------------------------------

    def _on_remote_message(self, message: Mapping[str, Any]) -> None:
        if not self._accept_remote:
            return
        with self._lock:
            if self._closed:
                return
            try:
                self._ingest_remote_unlocked(message)
            except Exception:
                self._remote_rejects += 1

    def _ingest_remote_unlocked(self, message: Mapping[str, Any]) -> None:
        if not isinstance(message, Mapping):
            raise RegistryValidationError("remote message must be a mapping")
        protocol = message.get("protocol_id")
        if protocol is not None and protocol != LIBP2P_DISCOVERY_PROTOCOL_ID:
            raise RegistryValidationError(
                f"unsupported discovery protocol_id {protocol!r}"
            )
        op = message.get("op")
        if op not in _KNOWN_OPS:
            raise RegistryValidationError(f"unknown discovery op {op!r}")

        receipt = message.get("now_ms")
        if not isinstance(receipt, int) or isinstance(receipt, bool) or receipt < 0:
            receipt = self._now(None)

        if op == OP_WITHDRAW:
            did = message.get("identity_did")
            if not isinstance(did, str) or not did.strip():
                # Fall back to advertisement identity when present.
                ad = message.get("advertisement")
                if isinstance(ad, Mapping):
                    did = identity_did(ad)
                else:
                    raise RegistryValidationError("withdraw requires identity_did")
            did = did.strip()
            self._records.pop(did, None)
            self._receipt_ms.pop(did, None)
            self._remote_ingests += 1
            return

        ad = message.get("advertisement")
        if not isinstance(ad, Mapping):
            raise RegistryValidationError("publish/refresh requires advertisement")
        try:
            normalized = validate_agent_advertisement(
                ad,
                require_signed=self._require_signed,
                now_ms=receipt,
                reject_stale=True,
                receipt_ms=receipt,
            )
        except RegistryStaleError:
            self._stale_rejects += 1
            raise
        except Exception as exc:
            from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
                RegistryUnsignedError,
            )

            if isinstance(exc, RegistryUnsignedError):
                self._unsigned_rejects += 1
            raise

        if op == OP_REFRESH:
            did = identity_did(normalized)
            if did not in self._records:
                # Remote refresh of an unknown DID is treated as publish so
                # late-joining peers still materialize the record.
                pass
        self._store(normalized, now_ms=receipt)
        self._remote_ingests += 1

    # ------------------------------------------------------------------
    # Registry@1 mutators
    # ------------------------------------------------------------------

    def publish(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate, store, and gossip an advertisement (upsert by DID)."""

        with self._lock:
            if self._closed:
                raise Libp2pDiscoveryError("discovery adapter is closed")
            current = self._now(now_ms)
            normalized = self._normalize(advertisement, now_ms=current)
            result = self._store(normalized, now_ms=current)
            self._publish_count += 1
            self._broadcast(OP_PUBLISH, advertisement=result, now_ms=current)
            return result

    def refresh(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Replace an existing local advertisement and gossip the refresh."""

        with self._lock:
            if self._closed:
                raise Libp2pDiscoveryError("discovery adapter is closed")
            current = self._now(now_ms)
            normalized = self._normalize(advertisement, now_ms=current)
            did = identity_did(normalized)
            if did not in self._records:
                raise RegistryNotFoundError(
                    f"cannot refresh unknown identity {did!r}"
                )
            result = self._store(normalized, now_ms=current)
            self._refresh_count += 1
            self._broadcast(OP_REFRESH, advertisement=result, now_ms=current)
            return result

    def withdraw(
        self,
        identity_did_value: str,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Remove a local advertisement and gossip the withdrawal."""

        if not isinstance(identity_did_value, str) or not identity_did_value.strip():
            raise RegistryValidationError("identity_did must be a non-empty string")
        did = identity_did_value.strip()
        with self._lock:
            if self._closed:
                raise Libp2pDiscoveryError("discovery adapter is closed")
            current = self._now(now_ms)
            existed = did in self._records
            self._records.pop(did, None)
            self._receipt_ms.pop(did, None)
            if existed:
                self._withdraw_count += 1
                self._broadcast(OP_WITHDRAW, identity=did, now_ms=current)
            return existed

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _fresh_copy(
        self,
        did: str,
        record: Mapping[str, Any],
        *,
        now_ms: int,
        include_stale: bool,
    ) -> Optional[dict[str, Any]]:
        receipt = self._receipt_ms.get(did)
        if not include_stale and is_stale(record, now_ms=now_ms, receipt_ms=receipt):
            return None
        return copy.deepcopy(dict(record))

    def lookup_by_identity(
        self,
        identity_did_value: str,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(identity_did_value, str) or not identity_did_value.strip():
            raise RegistryValidationError("identity_did must be a non-empty string")
        did = identity_did_value.strip()
        with self._lock:
            current = self._now(now_ms)
            record = self._records.get(did)
            if record is None:
                return None
            return self._fresh_copy(
                did, record, now_ms=current, include_stale=include_stale
            )

    def _iter_fresh(self, *, now_ms: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for did, record in self._records.items():
            copy_rec = self._fresh_copy(
                did, record, now_ms=now_ms, include_stale=False
            )
            if copy_rec is not None:
                out.append(copy_rec)
        return out

    def lookup_by_interface_cid(
        self,
        interface_cid: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(interface_cid, str) or not interface_cid.strip():
            raise RegistryValidationError("interface_cid must be a non-empty string")
        target = interface_cid.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in list(ad.get("interface_cids") or [])
            ]
            return sort_for_selection(matches)

    def lookup_by_semantic_capability(
        self,
        capability: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(capability, str) or not capability.strip():
            raise RegistryValidationError("capability must be a non-empty string")
        target = capability.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in semantic_capabilities_of(ad)
            ]
            return sort_for_selection(matches)

    def lookup_by_policy(
        self,
        policy_language: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(policy_language, str) or not policy_language.strip():
            raise RegistryValidationError("policy_language must be a non-empty string")
        target = policy_language.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in policy_languages_of(ad)
            ]
            return sort_for_selection(matches)

    def lookup_by_proof(
        self,
        proof_system: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(proof_system, str) or not proof_system.strip():
            raise RegistryValidationError("proof_system must be a non-empty string")
        target = proof_system.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in proof_systems_of(ad)
            ]
            return sort_for_selection(matches)

    def select(
        self,
        *,
        interface_cid: Optional[str] = None,
        semantic_capability: Optional[str] = None,
        policy_language: Optional[str] = None,
        proof_system: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        now_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            current = self._now(now_ms)
            if candidates is not None:
                pool = [dict(c) for c in candidates]
                filtered: list[dict[str, Any]] = []
                for ad in pool:
                    try:
                        if is_stale(ad, now_ms=current, receipt_ms=current):
                            continue
                    except RegistryValidationError:
                        continue
                    filtered.append(ad)
                pool = filtered
            else:
                pool = self._iter_fresh(now_ms=current)

            if interface_cid is not None:
                cid = interface_cid.strip()
                pool = [ad for ad in pool if cid in list(ad.get("interface_cids") or [])]
            if semantic_capability is not None:
                cap = semantic_capability.strip()
                pool = [ad for ad in pool if cap in semantic_capabilities_of(ad)]
            if policy_language is not None:
                pol = policy_language.strip()
                pool = [ad for ad in pool if pol in policy_languages_of(ad)]
            if proof_system is not None:
                proof = proof_system.strip()
                pool = [ad for ad in pool if proof in proof_systems_of(ad)]

            return select_advertisement(pool)

    def list_all(
        self,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            current = self._now(now_ms)
            out: list[dict[str, Any]] = []
            for did, record in self._records.items():
                copy_rec = self._fresh_copy(
                    did, record, now_ms=current, include_stale=include_stale
                )
                if copy_rec is not None:
                    out.append(copy_rec)
            out.sort(key=lambda ad: identity_did(ad))
            return out

    def stats(self) -> MutableMapping[str, Any]:
        with self._lock:
            return {
                "interface": self.interface,
                "family_interface": self.family_interface,
                "provider": self.provider_id,
                "execution_authority": False,
                "protocol_id": LIBP2P_DISCOVERY_PROTOCOL_ID,
                "peer_id": self._peer_id,
                "size": len(self._records),
                "publish_count": self._publish_count,
                "refresh_count": self._refresh_count,
                "withdraw_count": self._withdraw_count,
                "stale_rejects": self._stale_rejects,
                "unsigned_rejects": self._unsigned_rejects,
                "remote_ingests": self._remote_ingests,
                "remote_rejects": self._remote_rejects,
                "broadcasts": self._broadcasts,
                "require_signed": self._require_signed,
                "accept_remote": self._accept_remote,
                "closed": self._closed,
            }


def create_libp2p_discovery(
    *,
    transport: Optional[Libp2pDiscoveryTransport] = None,
    peer_id: Optional[str] = None,
    require_signed: bool = False,
    clock_ms: Optional[Callable[[], int]] = None,
    accept_remote: bool = True,
) -> Libp2pDiscovery:
    """Factory for :class:`Libp2pDiscovery`."""

    return Libp2pDiscovery(
        transport=transport,
        peer_id=peer_id,
        require_signed=require_signed,
        clock_ms=clock_ms,
        accept_remote=accept_remote,
    )


# Alias used by some plan wording ("libp2p adapter" / registry).
Libp2pRegistry = Libp2pDiscovery
create_libp2p_registry = create_libp2p_discovery


__all__ = [
    "LIBP2P_DISCOVERY_INTERFACE",
    "LIBP2P_DISCOVERY_PROTOCOL_ID",
    "LIBP2P_PROVIDER_ID",
    "OP_PUBLISH",
    "OP_REFRESH",
    "OP_WITHDRAW",
    "InMemoryLibp2pMesh",
    "Libp2pDiscovery",
    "Libp2pDiscoveryError",
    "Libp2pDiscoveryTransport",
    "Libp2pRegistry",
    "Libp2pTransportError",
    "create_libp2p_discovery",
    "create_libp2p_registry",
]
