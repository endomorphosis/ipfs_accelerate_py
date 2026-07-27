"""P2P service discovery and compact AI catalog advertisements.

Advertisements contain bounded discovery facts. Catalog records are fetched
separately, page by page, and pinned to the advertised content revision. The
legacy ``tools`` and ``metadata`` fields remain for wire compatibility, but new
catalog publishers never put model inventories in them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..model_catalog.identity import canonical_json_bytes
from ..model_catalog.schema import SCHEMA_VERSION
from ..model_catalog.security import (
    AdvertisementVerificationError,
    AdvertisementVerifier,
    CatalogAuthorizationPolicy,
    CatalogCapability,
    CatalogInputPolicy,
    ReplayCache,
    SecurityPolicyError,
)

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.service_registry")

SERVICE_NAMESPACE = "/mcppp/services/1.0.0"
CATALOG_PAGE_METHOD = "_mcppp_catalog_page"
CATALOG_ENDPOINT_PROTOCOL = "/mcp+p2p/catalog/1.0.0"
READVERTISE_INTERVAL = 60.0
SERVICE_TTL = 300.0
MAX_OPERATION_SUMMARY = 64
MAX_INTERFACE_CIDS = 32
MAX_PAGE_SIZE = 1_000
MAX_MULTIADDRS = 32
MAX_METADATA_ITEMS = 64
MAX_RECORD_TEXT_BYTES = 4_096
SIGNATURE_ALGORITHM = "hmac-sha256"

_NONCE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")


def _service_identity(service_name: str, issuer: str) -> str:
    payload = json.dumps(
        {
            "namespace": SERVICE_NAMESPACE,
            "service_name": service_name,
            "issuer": issuer,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "service_" + hashlib.sha256(payload).hexdigest()


def _catalog_operations(snapshot: Any) -> List[str]:
    operations = set()
    for collection_name in ("providers", "models", "deployments"):
        for record in tuple(getattr(snapshot, collection_name, ())):
            for capability in tuple(getattr(record, "capabilities", ())):
                operations.update(
                    getattr(item, "value", str(item))
                    for item in tuple(getattr(capability, "operations", ()))
                )
    for binding in tuple(getattr(snapshot, "bindings", ())):
        operations.update(
            getattr(item, "value", str(item))
            for item in tuple(getattr(binding, "operations", ()))
        )
    return sorted(operations)[:MAX_OPERATION_SUMMARY]


@dataclass
class ServiceRecord:
    """A signed service advertisement published to the P2P network."""

    service_name: str
    peer_id: str
    multiaddrs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)
    ttl: float = SERVICE_TTL
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    issuer: Optional[str] = None
    service_id: Optional[str] = None
    catalog_cid: Optional[str] = None
    catalog_revision: Optional[str] = None
    operation_summary: List[str] = field(default_factory=list)
    interface_cids: List[str] = field(default_factory=list)
    endpoint_protocol: Optional[str] = None
    issued_at: Optional[float] = None
    expires_at: Optional[float] = None
    schema_version: str = SCHEMA_VERSION
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    signature_algorithm: str = SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        for value, name in (
            (self.service_name, "service_name"),
            (self.peer_id, "peer_id"),
            (self.version, "version"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > MAX_RECORD_TEXT_BYTES
            ):
                raise ValueError("%s must be bounded non-empty text" % name)
        for value, name, maximum in (
            (self.multiaddrs, "multiaddrs", MAX_MULTIADDRS),
            (self.tools, "tools", MAX_OPERATION_SUMMARY),
            (self.operation_summary, "operation_summary", MAX_OPERATION_SUMMARY),
            (self.interface_cids, "interface_cids", MAX_INTERFACE_CIDS),
        ):
            if (
                not isinstance(value, list)
                or len(value) > maximum
                or any(
                    not isinstance(item, str)
                    or not item
                    or len(item.encode("utf-8")) > MAX_RECORD_TEXT_BYTES
                    for item in value
                )
            ):
                raise ValueError("%s must be a bounded text list" % name)
        if (
            not isinstance(self.metadata, Mapping)
            or len(self.metadata) > MAX_METADATA_ITEMS
        ):
            raise ValueError("metadata must be a bounded object")
        CatalogInputPolicy().validate_record(self.metadata)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported service record schema_version")
        if self.signature_algorithm != SIGNATURE_ALGORITHM:
            raise ValueError("unsupported service record signature algorithm")
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise ValueError("service record nonce is invalid")
        self.issuer = self.issuer or self.peer_id
        if (
            not isinstance(self.issuer, str)
            or not self.issuer
            or len(self.issuer.encode("utf-8")) > MAX_RECORD_TEXT_BYTES
        ):
            raise ValueError("issuer must be bounded non-empty text")
        self.service_id = self.service_id or _service_identity(
            self.service_name, self.issuer
        )
        if (
            not isinstance(self.service_id, str)
            or len(self.service_id.encode("utf-8")) > 256
        ):
            raise ValueError("service_id must be bounded text")
        self.issued_at = self.timestamp if self.issued_at is None else self.issued_at
        self.expires_at = (
            self.issued_at + self.ttl
            if self.expires_at is None
            else self.expires_at
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (self.issued_at, self.expires_at)
        ):
            raise ValueError("service record times must be finite numbers")
        self.timestamp = float(self.issued_at)
        self.ttl = max(0.0, float(self.expires_at) - self.timestamp)
        self.multiaddrs = list(self.multiaddrs)
        self.tools = sorted(set(self.tools))
        self.metadata = dict(self.metadata)
        self.operation_summary = sorted(set(self.operation_summary))
        self.interface_cids = sorted(set(self.interface_cids))

    @property
    def key(self) -> str:
        return f"{SERVICE_NAMESPACE}/{self.service_name}/{self.peer_id}"

    @property
    def is_catalog_advertisement(self) -> bool:
        values = (
            self.issuer,
            self.service_id,
            self.catalog_cid,
            self.catalog_revision,
            self.endpoint_protocol,
        )
        return all(isinstance(value, str) and bool(value) for value in values)

    def is_expired_at(self, now: Optional[float] = None) -> bool:
        selected = time.time() if now is None else float(now)
        return selected >= float(self.expires_at or 0.0)

    @property
    def is_expired(self) -> bool:
        return self.is_expired_at()

    def _unsigned_dict(self) -> Dict[str, Any]:
        """Return every advertisement field covered by the signature."""

        return {
            "service_name": self.service_name,
            "peer_id": self.peer_id,
            "multiaddrs": list(self.multiaddrs),
            "tools": sorted(self.tools),
            "version": self.version,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "issuer": self.issuer,
            "service_id": self.service_id,
            "catalog_cid": self.catalog_cid,
            "catalog_revision": self.catalog_revision,
            "operation_summary": list(self.operation_summary),
            "interface_cids": list(self.interface_cids),
            "endpoint_protocol": self.endpoint_protocol,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "schema_version": self.schema_version,
            "nonce": self.nonce,
            "signature_algorithm": self.signature_algorithm,
        }

    def signing_payload(self) -> bytes:
        return canonical_json_bytes(self._unsigned_dict())

    def sign(
        self, key: Optional[bytes | str] = None, *, rotate_nonce: bool = False
    ) -> None:
        """Sign using the existing HMAC compatibility mechanism."""

        if rotate_nonce:
            self.nonce = secrets.token_urlsafe(24)
        signing_key = (
            key.encode("utf-8")
            if isinstance(key, str)
            else key or self.peer_id.encode("utf-8")
        )
        if not signing_key:
            raise ValueError("signing key must not be empty")
        self.signature = hmac.new(
            signing_key, self.signing_payload(), hashlib.sha256
        ).hexdigest()

    def verify_signature(self, key: Optional[bytes | str] = None) -> bool:
        if not isinstance(self.signature, str) or not _SIGNATURE.fullmatch(
            self.signature
        ):
            return False
        signing_key = (
            key.encode("utf-8")
            if isinstance(key, str)
            else key or self.peer_id.encode("utf-8")
        )
        try:
            expected = hmac.new(
                signing_key, self.signing_payload(), hashlib.sha256
            ).hexdigest()
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        result = self._unsigned_dict()
        result["signature"] = self.signature
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ServiceRecord":
        if not isinstance(data, Mapping):
            raise TypeError("service record must be an object")
        CatalogInputPolicy().validate_record(data)
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("service record contains unknown fields")
        for field_name in ("multiaddrs", "tools", "operation_summary", "interface_cids"):
            value = data.get(field_name, [])
            if not isinstance(value, list):
                raise TypeError("%s must be a list" % field_name)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            service_name=data.get("service_name", ""),
            peer_id=data.get("peer_id", ""),
            multiaddrs=list(data.get("multiaddrs", [])),
            tools=list(data.get("tools", [])),
            version=data.get("version", "1.0.0"),
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl", SERVICE_TTL),
            metadata=dict(data.get("metadata", {})),
            signature=data.get("signature"),
            issuer=data.get("issuer"),
            service_id=data.get("service_id"),
            catalog_cid=data.get("catalog_cid"),
            catalog_revision=data.get("catalog_revision"),
            operation_summary=list(data.get("operation_summary", [])),
            interface_cids=list(data.get("interface_cids", [])),
            endpoint_protocol=data.get("endpoint_protocol"),
            issued_at=data.get("issued_at"),
            expires_at=data.get("expires_at"),
            schema_version=data.get("schema_version", ""),
            nonce=data.get("nonce", ""),
            signature_algorithm=data.get("signature_algorithm", ""),
        )


class ServiceRegistry:
    """Thread-safe local and discovered service registry."""

    MAX_REMOTE_RECORDS_PER_SERVICE = 200

    def __init__(
        self,
        *,
        trusted_issuers: Optional[Mapping[str, bytes | str]] = None,
        authorization_policy: Optional[CatalogAuthorizationPolicy] = None,
        max_clock_skew: float = 30.0,
        replay_window: float = 600.0,
        max_advertisement_lifetime: float = 600.0,
    ):
        self._lock = threading.Lock()
        self._local_records: Dict[str, ServiceRecord] = {}
        self._remote_records: Dict[str, Dict[str, ServiceRecord]] = {}
        self._change_callbacks: Dict[str, Callable] = {}
        self._catalog_providers: Dict[str, Callable[[], Any]] = {}
        self._catalog_snapshots: Dict[str, Any] = {}
        # ``None`` retains authenticated-peer compatibility: the peer identity
        # is the trust anchor for its own announcement.  Passing a mapping,
        # including an empty one, enables an explicit authoritative trust store.
        self._trusted_issuers = (
            None if trusted_issuers is None else dict(trusted_issuers)
        )
        self._authorization_policy = authorization_policy
        self._max_clock_skew = max_clock_skew
        self._replay_window = replay_window
        self._max_advertisement_lifetime = max_advertisement_lifetime
        self._replay_cache = ReplayCache()

    @staticmethod
    def _catalog_resource(service_id: str) -> str:
        return "catalog:%s" % service_id

    def require_capability(
        self,
        actor: str,
        resource: str,
        capability: str | CatalogCapability,
    ) -> None:
        """Enforce an exact configured capability, if policy is enabled."""

        if self._authorization_policy is None:
            return
        ability = (
            capability.value
            if isinstance(capability, CatalogCapability)
            else capability
        )
        self._authorization_policy.require(actor, resource, ability)

    def _advertisement_verifier(
        self, record: ServiceRecord, sender_peer_id: str
    ) -> AdvertisementVerifier:
        if self._trusted_issuers is None:
            # The transport-authenticated sender may speak only for itself.
            if (
                not sender_peer_id
                or record.issuer != sender_peer_id
                or record.peer_id != sender_peer_id
            ):
                raise AdvertisementVerificationError(
                    "issuer_untrusted",
                    "advertisement issuer is not the authenticated sender",
                )
            keys: Mapping[str, bytes | str] = {
                sender_peer_id: sender_peer_id.encode("utf-8")
            }
        else:
            keys = self._trusted_issuers
        return AdvertisementVerifier(
            keys,
            max_clock_skew=self._max_clock_skew,
            replay_window=self._replay_window,
            max_lifetime=self._max_advertisement_lifetime,
            replay_cache=self._replay_cache,
        )

    def _verify_remote(
        self,
        record: ServiceRecord,
        *,
        sender_peer_id: str,
        now: Optional[float] = None,
        consume_nonce: bool = True,
    ) -> None:
        if not record.is_catalog_advertisement:
            raise AdvertisementVerificationError(
                "invalid_catalog_advertisement",
                "catalog advertisement is incomplete",
            )
        expected_service_id = _service_identity(record.service_name, record.issuer or "")
        if record.service_id != expected_service_id:
            raise AdvertisementVerificationError(
                "service_identity_invalid",
                "service identity does not match signed fields",
            )
        self._advertisement_verifier(record, sender_peer_id).verify(
            record,
            now=now,
            consume_nonce=consume_nonce,
        )

    def register_local(
        self,
        record: ServiceRecord,
        *,
        catalog_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        with self._lock:
            self._local_records[record.service_name] = record
            if catalog_provider is not None:
                self._catalog_providers[record.service_name] = catalog_provider
        if catalog_provider is not None:
            self.refresh_local_catalogs((record.service_name,))
        logger.info(
            "Registered local service: %s (%d operations)",
            record.service_name,
            len(record.operation_summary or record.tools),
        )

    def unregister_local(
        self, service_name: str, *, peer_id: Optional[str] = None
    ) -> bool:
        with self._lock:
            record = self._local_records.get(service_name)
            if record is None or (peer_id is not None and record.peer_id != peer_id):
                return False
            del self._local_records[service_name]
            self._catalog_providers.pop(service_name, None)
            self._catalog_snapshots.pop(service_name, None)
        self._notify_change("remove", record)
        return True

    def update_local_catalog(
        self,
        service_name: str,
        snapshot: Any,
        *,
        now: Optional[float] = None,
    ) -> bool:
        """Atomically pin a local record to a snapshot revision."""

        revision = getattr(snapshot, "revision", None)
        if not isinstance(revision, str) or not revision:
            raise ValueError("catalog snapshot must expose a content revision")
        selected_now = time.time() if now is None else float(now)
        with self._lock:
            record = self._local_records.get(service_name)
            if record is None:
                raise KeyError("unknown local service: %s" % service_name)
            changed = record.catalog_revision != revision
            record.catalog_cid = revision
            record.catalog_revision = revision
            if not record.operation_summary:
                record.operation_summary = _catalog_operations(snapshot)
            record.endpoint_protocol = (
                record.endpoint_protocol or CATALOG_ENDPOINT_PROTOCOL
            )
            lifetime = max(record.ttl, SERVICE_TTL)
            record.issued_at = selected_now
            record.expires_at = selected_now + lifetime
            record.timestamp = selected_now
            record.ttl = lifetime
            record.sign(rotate_nonce=True)
            self._catalog_snapshots[service_name] = snapshot
        if changed:
            self._notify_change("catalog_update", record)
        return changed

    def refresh_local_catalogs(
        self,
        service_names: Optional[Tuple[str, ...]] = None,
        *,
        now: Optional[float] = None,
    ) -> Tuple[str, ...]:
        with self._lock:
            names = (
                tuple(sorted(self._catalog_providers))
                if service_names is None
                else tuple(service_names)
            )
            providers = {
                name: self._catalog_providers[name]
                for name in names
                if name in self._catalog_providers
            }
        changed = []
        for name in sorted(providers):
            try:
                snapshot = providers[name]()
                if self.update_local_catalog(name, snapshot, now=now):
                    changed.append(name)
            except Exception as exc:
                logger.debug("Catalog refresh failed for %s: %s", name, exc)
        return tuple(changed)

    def add_remote(
        self,
        record: ServiceRecord,
        *,
        sender_peer_id: Optional[str] = None,
        now: Optional[float] = None,
        _verified: bool = False,
    ) -> bool:
        """Admit a remote record after catalog security verification.

        Direct callers are treated as the authenticated peer represented by
        ``record.peer_id`` unless they provide the transport identity.
        """

        selected_now = time.time() if now is None else float(now)
        if record.is_expired_at(selected_now):
            return False
        catalog_candidate = (
            record.catalog_cid is not None or record.catalog_revision is not None
        )
        if catalog_candidate and not _verified:
            try:
                self._verify_remote(
                    record,
                    sender_peer_id=sender_peer_id or record.peer_id,
                    now=selected_now,
                )
            except SecurityPolicyError:
                return False
        with self._lock:
            bucket = self._remote_records.setdefault(record.service_name, {})
            previous = bucket.get(record.peer_id)
            if previous is not None and float(previous.issued_at or 0) > float(
                record.issued_at or 0
            ):
                return False
            if previous is not None and previous.to_dict() == record.to_dict():
                return False
            if previous is None and len(bucket) >= self.MAX_REMOTE_RECORDS_PER_SERVICE:
                oldest_pid = min(
                    bucket, key=lambda pid: float(bucket[pid].issued_at or 0)
                )
                del bucket[oldest_pid]
            bucket[record.peer_id] = record
        self._notify_change("add" if previous is None else "update", record)
        return True

    def remove_remote(
        self, peer_id: str, *, service_name: Optional[str] = None
    ) -> int:
        removed: List[ServiceRecord] = []
        with self._lock:
            names = (
                (service_name,)
                if service_name is not None
                else tuple(self._remote_records)
            )
            for name in names:
                bucket = self._remote_records.get(name)
                if not bucket:
                    continue
                record = bucket.pop(peer_id, None)
                if record is not None:
                    removed.append(record)
                if not bucket:
                    self._remote_records.pop(name, None)
        for record in removed:
            self._notify_change("remove", record)
        return len(removed)

    def on_change(self, callback_id: str, callback: Callable) -> None:
        with self._lock:
            self._change_callbacks[callback_id] = callback

    def remove_callback(self, callback_id: str) -> None:
        with self._lock:
            self._change_callbacks.pop(callback_id, None)

    def _notify_change(self, event_type: str, record: ServiceRecord) -> None:
        with self._lock:
            callbacks = list(self._change_callbacks.values())
        for callback in callbacks:
            try:
                callback(event_type, record)
            except Exception as exc:
                logger.debug("Service change callback error: %s", exc)

    def get_services(
        self, service_name: Optional[str] = None
    ) -> List[ServiceRecord]:
        with self._lock:
            buckets = (
                (self._remote_records.get(service_name, {}),)
                if service_name
                else tuple(self._remote_records.values())
            )
            return [
                record
                for bucket in buckets
                for record in bucket.values()
                if not record.is_expired
            ]

    def get_local(self, service_name: str) -> Optional[ServiceRecord]:
        with self._lock:
            return self._local_records.get(service_name)

    def get_peers_for_tool(self, tool_name: str) -> List[ServiceRecord]:
        return [
            record
            for record in self.get_services()
            if tool_name in record.tools or tool_name in record.operation_summary
        ]

    def cleanup_stale(self, *, now: Optional[float] = None) -> int:
        removed: List[ServiceRecord] = []
        with self._lock:
            for service_name in list(self._remote_records):
                bucket = self._remote_records[service_name]
                for peer_id, record in list(bucket.items()):
                    if record.is_expired_at(now):
                        removed.append(bucket.pop(peer_id))
                if not bucket:
                    del self._remote_records[service_name]
        for record in removed:
            self._notify_change("expire", record)
        return len(removed)

    async def advertise_once(
        self, p2p_node: Any, *, now: Optional[float] = None
    ) -> Tuple[ServiceRecord, ...]:
        selected_now = time.time() if now is None else float(now)
        self.refresh_local_catalogs(now=selected_now)
        with self._lock:
            records = tuple(self._local_records.values())
        for record in records:
            if record.catalog_revision is None:
                record.issued_at = selected_now
                record.expires_at = selected_now + SERVICE_TTL
                record.timestamp = selected_now
                record.ttl = SERVICE_TTL
                record.sign(rotate_nonce=True)
            for peer_id in list(p2p_node._peers.keys()):
                try:
                    await p2p_node.call_tool(
                        peer_id,
                        "_mcppp_service_announce",
                        {"record": record.to_dict()},
                        timeout=5.0,
                    )
                except Exception:
                    pass
        self.cleanup_stale(now=selected_now)
        return records

    async def advertise_loop(self, p2p_node: Any, nursery: Any = None) -> None:
        import trio

        while True:
            try:
                await self.advertise_once(p2p_node)
                await trio.sleep(READVERTISE_INTERVAL)
            except trio.Cancelled:
                break
            except Exception as exc:
                logger.debug("Service advertise error: %s", exc)
                await trio.sleep(READVERTISE_INTERVAL)

    async def catalog_watch_loop(
        self, p2p_node: Any, *, poll_interval: float = 1.0
    ) -> None:
        import trio

        interval = max(0.05, min(float(poll_interval), READVERTISE_INTERVAL))
        while True:
            try:
                await trio.sleep(interval)
                if self.refresh_local_catalogs():
                    await self.advertise_once(p2p_node)
            except trio.Cancelled:
                break
            except Exception as exc:
                logger.debug("Catalog watch error: %s", exc)

    def handle_catalog_page(
        self, params: Mapping[str, Any], sender_peer_id: str = ""
    ) -> Dict[str, Any]:
        from ipfs_accelerate_py.model_catalog.snapshot import paginate_snapshot

        try:
            CatalogInputPolicy().validate_record(params)
        except SecurityPolicyError as exc:
            raise ValueError(exc.code) from None
        service_name = params.get("service_name", "")
        revision = params.get("catalog_revision", "")
        record_type = params.get("record_type", "")
        cursor = params.get("cursor")
        limit = params.get("limit", 100)
        if (
            not isinstance(service_name, str)
            or not isinstance(revision, str)
            or record_type
            not in {"providers", "models", "deployments", "bindings"}
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PAGE_SIZE
        ):
            raise ValueError("invalid catalog page request")
        with self._lock:
            record = self._local_records.get(service_name)
            snapshot = self._catalog_snapshots.get(service_name)
        if record is None or snapshot is None:
            raise KeyError("catalog service is unavailable")
        self.require_capability(
            sender_peer_id,
            self._catalog_resource(record.service_id or ""),
            CatalogCapability.READ,
        )
        if revision != record.catalog_revision or revision != snapshot.revision:
            raise ValueError("catalog revision is stale or unavailable")
        return paginate_snapshot(
            snapshot, record_type, limit=limit, cursor=cursor
        ).to_dict()

    def handle_announce(
        self, params: Mapping[str, Any], sender_peer_id: str = ""
    ) -> Dict[str, Any]:
        try:
            CatalogInputPolicy().validate_record(params)
        except SecurityPolicyError as exc:
            return exc.to_dict()
        record_data = params.get("record", {})
        if not isinstance(record_data, Mapping) or not record_data:
            return {"status": "rejected", "reason": "empty_record"}
        try:
            record = ServiceRecord.from_dict(record_data)
        except (TypeError, ValueError):
            return {"status": "rejected", "reason": "malformed_record"}
        if sender_peer_id and record.peer_id != sender_peer_id:
            return {"status": "rejected", "reason": "peer_id_mismatch"}
        selected_now = time.time()
        if record.is_expired_at(selected_now):
            return {"status": "rejected", "reason": "expired"}
        require_signature = (
            record.catalog_cid is not None
            or record.catalog_revision is not None
            or os.environ.get("MCPPP_REQUIRE_SERVICE_SIGNATURES", "0") == "1"
        )
        catalog_candidate = (
            record.catalog_cid is not None or record.catalog_revision is not None
        )
        if catalog_candidate and not record.is_catalog_advertisement:
            return {"status": "rejected", "reason": "invalid_catalog_advertisement"}
        if catalog_candidate:
            try:
                self._verify_remote(
                    record,
                    sender_peer_id=sender_peer_id or record.peer_id,
                    now=selected_now,
                )
            except SecurityPolicyError as exc:
                return exc.to_dict()
        elif require_signature and not record.verify_signature():
            return {"status": "rejected", "reason": "invalid_signature"}
        if not self.add_remote(
            record,
            sender_peer_id=sender_peer_id or record.peer_id,
            now=selected_now,
            _verified=catalog_candidate,
        ):
            return {"status": "rejected", "reason": "stale_or_duplicate"}
        return {
            "status": "accepted",
            "service": record.service_name,
            "catalog_revision": record.catalog_revision,
        }

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "local_services": {
                    key: value.to_dict()
                    for key, value in self._local_records.items()
                },
                "remote_services": {
                    service: [
                        record.to_dict()
                        for record in records.values()
                        if not record.is_expired
                    ]
                    for service, records in self._remote_records.items()
                },
                "total_remote_peers": sum(
                    sum(not record.is_expired for record in records.values())
                    for records in self._remote_records.values()
                ),
            }


_REGISTRY: Optional[ServiceRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_service_registry() -> ServiceRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = ServiceRegistry()
    return _REGISTRY


__all__ = [
    "CATALOG_ENDPOINT_PROTOCOL",
    "CATALOG_PAGE_METHOD",
    "READVERTISE_INTERVAL",
    "SERVICE_NAMESPACE",
    "SERVICE_TTL",
    "ServiceRecord",
    "ServiceRegistry",
    "get_service_registry",
]
