"""Bounded, transport-injected federation of peer catalog snapshots.

This module never opens a socket, resolves a host, follows a URL, or discovers
credentials. A caller must inject a transport which explicitly authorizes each
advertisement and returns deterministic revision-bound pages.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from ..schema import (
    CatalogSnapshot,
    DeploymentDescriptor,
    ModelDescriptor,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ..snapshot import CatalogPage
from .static import CatalogSourceResult, SourceDiagnostic, SourceMetadata

DEFAULT_PEER_PRECEDENCE = -100
DEFAULT_PAGE_SIZE = 100
MAX_PEERS = 64
MAX_PAGES_PER_PEER = 1_024
MAX_PEER_RECORDS = 10_000
_RECORD_TYPES = ("providers", "models", "deployments", "bindings")
_TYPE_BY_RECORD = {
    "providers": ProviderDescriptor,
    "models": ModelDescriptor,
    "deployments": DeploymentDescriptor,
    "bindings": RouterBinding,
}
_ID_BY_RECORD = {
    "providers": "provider_id",
    "models": "model_id",
    "deployments": "deployment_id",
    "bindings": "binding_id",
}


class PeerCatalogError(ValueError):
    """A peer advertisement or fetched page failed closed."""


class PeerCatalogTransport(Protocol):
    """Minimal injected transport used by :class:`PeerCatalogSource`."""

    authorized: bool

    def fetch_catalog_page(
        self,
        advertisement: Any,
        *,
        record_type: str,
        cursor: Optional[str],
        limit: int,
    ) -> Mapping[str, Any]:
        """Fetch one already-authorized peer page."""


@dataclass(frozen=True)
class _CachedPeer:
    advertisement_key: Tuple[str, str, float]
    snapshot: CatalogSnapshot


def _canonical_name(value: Any, field: str, maximum: int = 128) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be text" % field)
    value = re.sub(r"[^a-z0-9._/-]+", "-", value.strip().casefold())
    value = re.sub(r"/+", "/", value).strip("-._/")
    value = re.sub(r"\.{2,}", ".", value)
    if (
        not value
        or len(value.encode("utf-8")) > maximum
        or "//" in value
        or ".." in value
    ):
        raise ValueError("%s must be a bounded canonical name" % field)
    return value


def _rfc3339(value: float) -> str:
    return (
        datetime.fromtimestamp(float(value), timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _advertisement_key(record: Any) -> Tuple[str, str, float]:
    return (
        str(record.service_id),
        str(record.catalog_revision),
        float(record.issued_at),
    )


class PeerCatalogSource:
    """Federate complete verified snapshots from advertised peers.

    ``load`` is side-effect free and returns the last published generation.
    ``refresh`` is the only operation that calls the injected transport, so an
    aggregate catalog must authorize this adapter as a side-effecting source.
    """

    side_effecting = True

    def __init__(
        self,
        advertisements: Any,
        transport: PeerCatalogTransport,
        *,
        trust_domain: str,
        service_name: str = "ipfs-accelerate-mcp",
        source: Optional[str] = None,
        precedence: int = DEFAULT_PEER_PRECEDENCE,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_peers: int = MAX_PEERS,
        max_pages_per_peer: int = MAX_PAGES_PER_PEER,
        max_records: int = MAX_PEER_RECORDS,
    ) -> None:
        self.trust_domain = _canonical_name(trust_domain, "trust_domain", 64)
        self._domain_token = hashlib.sha256(
            self.trust_domain.encode("utf-8")
        ).hexdigest()[:16]
        self.source = _canonical_name(
            source or "peers/%s" % self._domain_token, "source"
        )
        self.service_name = _canonical_name(service_name, "service_name")
        if (
            isinstance(precedence, bool)
            or not isinstance(precedence, int)
            or not -1_000_000 <= precedence <= 0
        ):
            raise ValueError("peer precedence must be between -1000000 and 0")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 1_000
        ):
            raise ValueError("page_size must be between 1 and 1000")
        for value, name, maximum in (
            (max_peers, "max_peers", MAX_PEERS),
            (max_pages_per_peer, "max_pages_per_peer", MAX_PAGES_PER_PEER),
            (max_records, "max_records", MAX_PEER_RECORDS),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError("%s must be between 1 and %d" % (name, maximum))
        self.precedence = precedence
        self.page_size = page_size
        self.max_peers = max_peers
        self.max_pages_per_peer = max_pages_per_peer
        self.max_records = max_records
        self._advertisements = advertisements
        self._transport = transport
        self._peer_cache: Dict[Tuple[str, str], _CachedPeer] = {}
        self._last_result = self._result(CatalogSnapshot(), ())

    def _result(
        self,
        snapshot: CatalogSnapshot,
        diagnostics: Sequence[SourceDiagnostic],
    ) -> CatalogSourceResult:
        return CatalogSourceResult(
            snapshot=snapshot,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=snapshot.revision,
            ),
            diagnostics=tuple(diagnostics),
        )

    def load(self) -> CatalogSourceResult:
        """Return cached records without contacting any peer."""

        return self._last_result

    snapshot = load
    read = load

    def _listed_advertisements(self) -> Tuple[Any, ...]:
        value = self._advertisements
        if callable(value):
            value = value()
        elif callable(getattr(value, "get_services", None)):
            value = value.get_services(self.service_name)
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
            value, Iterable
        ):
            raise PeerCatalogError("advertisements must be a bounded iterable")
        records = tuple(value)
        if len(records) > self.max_peers * 4:
            raise PeerCatalogError("advertisement input exceeds peer bound")
        return records

    @staticmethod
    def _is_expired(record: Any) -> bool:
        checker = getattr(record, "is_expired", False)
        return bool(checker() if callable(checker) else checker)

    def _select_advertisements(self) -> Tuple[Any, ...]:
        selected: Dict[Tuple[str, str], Any] = {}
        for record in self._listed_advertisements():
            if (
                getattr(record, "service_name", None) != self.service_name
                or not getattr(record, "is_catalog_advertisement", False)
                or self._is_expired(record)
                or not callable(getattr(record, "verify_signature", None))
                or not record.verify_signature()
                or record.catalog_cid != record.catalog_revision
            ):
                continue
            key = (str(record.issuer), str(record.service_id))
            previous = selected.get(key)
            if previous is None or (
                float(record.issued_at),
                str(record.catalog_revision),
            ) > (
                float(previous.issued_at),
                str(previous.catalog_revision),
            ):
                selected[key] = record
        if len(selected) > self.max_peers:
            selected = dict(
                sorted(
                    selected.items(),
                    key=lambda item: (float(item[1].issued_at), item[0]),
                    reverse=True,
                )[: self.max_peers]
            )
        return tuple(selected[key] for key in sorted(selected))

    def _authorize(self, record: Any) -> None:
        decision = None
        for name in ("authorized_for", "is_authorized"):
            checker = getattr(self._transport, name, None)
            if callable(checker):
                try:
                    decision = checker(record, "catalog.read")
                except TypeError:
                    decision = checker(record)
                break
        if decision is None:
            decision = getattr(self._transport, "authorized", None)
        if decision is not True:
            raise PermissionError("transport did not authorize catalog.read")

    def _transport_page(
        self, record: Any, record_type: str, cursor: Optional[str]
    ) -> Any:
        self._authorize(record)
        method = getattr(self._transport, "fetch_catalog_page", None)
        if not callable(method):
            method = getattr(self._transport, "fetch_page", None)
        if not callable(method):
            raise TypeError("authorized transport has no catalog page operation")
        return method(
            record,
            record_type=record_type,
            cursor=cursor,
            limit=self.page_size,
        )

    def _fetch_record_type(self, record: Any, record_type: str) -> Tuple[Any, ...]:
        expected_revision = record.catalog_revision
        expected_total: Optional[int] = None
        cursor: Optional[str] = None
        seen_cursors = set()
        items = []
        for _page_number in range(self.max_pages_per_peer):
            raw_page = self._transport_page(record, record_type, cursor)
            page = raw_page.to_dict() if isinstance(raw_page, CatalogPage) else raw_page
            if not isinstance(page, Mapping):
                raise PeerCatalogError("peer page must be an object")
            page_revision = page.get(
                "snapshot_revision", page.get("catalog_revision")
            )
            if page_revision != expected_revision:
                raise PeerCatalogError("peer page revision does not match advertisement")
            if page.get("record_type") != record_type:
                raise PeerCatalogError("peer page record type does not match request")
            total = page.get("total")
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise PeerCatalogError("peer page total is invalid")
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise PeerCatalogError("peer page total changed during pagination")
            raw_items = page.get("items")
            if (
                isinstance(raw_items, (str, bytes, Mapping))
                or not isinstance(raw_items, Sequence)
                or len(raw_items) > self.page_size
            ):
                raise PeerCatalogError("peer page items are invalid or excessive")
            parsed = [
                item
                if isinstance(item, _TYPE_BY_RECORD[record_type])
                else _TYPE_BY_RECORD[record_type].from_dict(item)
                for item in raw_items
            ]
            items.extend(parsed)
            if len(items) > self.max_records or len(items) > total:
                raise PeerCatalogError("peer records exceed advertised bounds")
            following = page.get("next_cursor")
            if following is None:
                if len(items) != total:
                    raise PeerCatalogError("peer pagination ended before total")
                return tuple(items)
            if (
                not isinstance(following, str)
                or not following
                or following == cursor
                or following in seen_cursors
                or not raw_items
            ):
                raise PeerCatalogError("peer pagination cursor did not progress")
            seen_cursors.add(following)
            cursor = following
        raise PeerCatalogError("peer pagination exceeds page bound")

    def _fetch_snapshot(self, record: Any) -> CatalogSnapshot:
        collections = {
            record_type: self._fetch_record_type(record, record_type)
            for record_type in _RECORD_TYPES
        }
        if sum(len(items) for items in collections.values()) > self.max_records:
            raise PeerCatalogError("peer snapshot exceeds aggregate record bound")
        snapshot = CatalogSnapshot(**collections)
        if (
            snapshot.revision != record.catalog_revision
            or snapshot.cid != record.catalog_cid
        ):
            raise PeerCatalogError(
                "canonical catalog content does not match advertised CID"
            )
        return snapshot

    def _peer_provenance(self, record: Any, original_id: str) -> Provenance:
        issuer = "%s:%s" % (self.trust_domain, record.issuer)
        if len(issuer.encode("utf-8")) > 512:
            issuer = "%s:%s" % (
                self.trust_domain,
                hashlib.sha256(str(record.issuer).encode("utf-8")).hexdigest(),
            )
        source_record_id = "%s:%s:%s" % (
            record.service_id,
            record.catalog_revision,
            original_id,
        )
        if len(source_record_id.encode("utf-8")) > 512:
            source_record_id = hashlib.sha256(
                source_record_id.encode("utf-8")
            ).hexdigest()
        return Provenance(
            source=self.source,
            source_record_id=source_record_id,
            observed_at=_rfc3339(record.issued_at),
            expires_at=_rfc3339(record.expires_at),
            issuer=issuer,
        )

    def _isolate_snapshot(
        self, snapshot: CatalogSnapshot, record: Any
    ) -> CatalogSnapshot:
        """Verify first, then rebuild every identity under the trust domain."""

        providers = []
        provider_ids: Dict[str, str] = {}
        for provider in snapshot.providers:
            remote_id = provider.provider_id
            token = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:24]
            isolated = ProviderDescriptor(
                name="federated/%s/%s" % (self._domain_token, token),
                display_name=provider.display_name or provider.name,
                aliases=(),
                description=provider.description,
                website_uri=provider.website_uri,
                documentation_uri=provider.documentation_uri,
                capabilities=provider.capabilities,
                lifecycle=provider.lifecycle,
                state=provider.state,
                provenance=(self._peer_provenance(record, remote_id),),
                labels=provider.labels,
            )
            providers.append(isolated)
            provider_ids[remote_id] = isolated.provider_id

        models = []
        model_ids: Dict[str, str] = {}
        for model in snapshot.models:
            if model.provider_id not in provider_ids:
                raise PeerCatalogError("peer model references an unknown provider")
            isolated = ModelDescriptor(
                provider_id=provider_ids[model.provider_id],
                name=model.name,
                display_name=model.display_name,
                aliases=(),
                description=model.description,
                architecture=model.architecture,
                capabilities=model.capabilities,
                lifecycle=model.lifecycle,
                state=model.state,
                provenance=(self._peer_provenance(record, model.model_id),),
                labels=model.labels,
            )
            models.append(isolated)
            model_ids[model.model_id] = isolated.model_id

        deployments = []
        deployment_ids: Dict[str, str] = {}
        for deployment in snapshot.deployments:
            if deployment.provider_id not in provider_ids:
                raise PeerCatalogError("peer deployment references an unknown provider")
            mapped_model = None
            if deployment.model_id is not None:
                if deployment.model_id not in model_ids:
                    raise PeerCatalogError("peer deployment references an unknown model")
                mapped_model = model_ids[deployment.model_id]
            isolated = DeploymentDescriptor(
                provider_id=provider_ids[deployment.provider_id],
                model_id=mapped_model,
                name=deployment.name,
                endpoint_uri=deployment.endpoint_uri,
                capabilities=deployment.capabilities,
                lifecycle=deployment.lifecycle,
                state=deployment.state,
                created_at=deployment.created_at,
                updated_at=deployment.updated_at,
                provenance=(
                    self._peer_provenance(record, deployment.deployment_id),
                ),
                labels=deployment.labels,
            )
            deployments.append(isolated)
            deployment_ids[deployment.deployment_id] = isolated.deployment_id

        bindings = []
        for binding in snapshot.bindings:
            if binding.provider_id not in provider_ids:
                raise PeerCatalogError("peer binding references an unknown provider")
            mapped_model = (
                None
                if binding.model_id is None
                else model_ids.get(binding.model_id)
            )
            mapped_deployment = (
                None
                if binding.deployment_id is None
                else deployment_ids.get(binding.deployment_id)
            )
            if binding.model_id is not None and mapped_model is None:
                raise PeerCatalogError("peer binding references an unknown model")
            if binding.deployment_id is not None and mapped_deployment is None:
                raise PeerCatalogError("peer binding references an unknown deployment")
            router_token = hashlib.sha256(
                binding.router.encode("utf-8")
            ).hexdigest()[:12]
            bindings.append(
                RouterBinding(
                    router="peer_%s_%s"
                    % (self._domain_token[:12], router_token),
                    provider_id=provider_ids[binding.provider_id],
                    model_id=mapped_model,
                    deployment_id=mapped_deployment,
                    operations=binding.operations,
                    priority=binding.priority,
                    state=binding.state,
                    provenance=(
                        self._peer_provenance(record, binding.binding_id),
                    ),
                    labels=binding.labels,
                )
            )
        return CatalogSnapshot(
            providers=tuple(providers),
            models=tuple(models),
            deployments=tuple(deployments),
            bindings=tuple(bindings),
        )

    @staticmethod
    def _combine(peers: Sequence[Tuple[Any, CatalogSnapshot]]) -> CatalogSnapshot:
        collections: Dict[str, Dict[str, Any]] = {
            name: {} for name in _RECORD_TYPES
        }
        for advertisement, snapshot in sorted(
            peers,
            key=lambda item: (
                float(item[0].issued_at),
                str(item[0].issuer),
                str(item[0].service_id),
            ),
        ):
            for record_type in _RECORD_TYPES:
                id_field = _ID_BY_RECORD[record_type]
                for item in getattr(snapshot, record_type):
                    collections[record_type][getattr(item, id_field)] = item
        return CatalogSnapshot(
            **{
                name: tuple(values.values())
                for name, values in collections.items()
            }
        )

    def refresh(self) -> CatalogSourceResult:
        """Fetch a complete generation, isolating failure to each peer."""

        advertisements = self._select_advertisements()
        active_keys = {
            (str(record.issuer), str(record.service_id))
            for record in advertisements
        }
        self._peer_cache = {
            key: value
            for key, value in self._peer_cache.items()
            if key in active_keys
        }
        diagnostics = []
        accepted = []
        for record in advertisements:
            key = (str(record.issuer), str(record.service_id))
            advertised_key = _advertisement_key(record)
            cached = self._peer_cache.get(key)
            if cached is not None and cached.advertisement_key == advertised_key:
                accepted.append((record, cached.snapshot))
                continue
            try:
                remote = self._fetch_snapshot(record)
                isolated = self._isolate_snapshot(remote, record)
                self._peer_cache[key] = _CachedPeer(
                    advertisement_key=advertised_key,
                    snapshot=isolated,
                )
                accepted.append((record, isolated))
            except Exception as exc:
                # Preserve a verified generation only while its exact
                # advertisement remains current; never extend its lease.
                if (
                    cached is not None
                    and cached.advertisement_key == advertised_key
                ):
                    accepted.append((record, cached.snapshot))
                diagnostics.append(
                    SourceDiagnostic(
                        index=None,
                        code="peer_fetch_failed",
                        message=("peer catalog rejected: %s" % type(exc).__name__)[:256],
                        source_record_id=str(record.service_id)[:128],
                    )
                )
        combined = self._combine(accepted)
        self._last_result = self._result(combined, diagnostics)
        return self._last_result

    fetch = refresh

    def disconnect(self, peer_id: str) -> bool:
        """Drop cached state for one peer or issuer."""

        keys = [
            key
            for key in self._peer_cache
            if key[0] == peer_id
            or any(
                str(record.peer_id) == peer_id
                and (str(record.issuer), str(record.service_id)) == key
                for record in self._select_advertisements()
            )
        ]
        for key in keys:
            del self._peer_cache[key]
        if not keys:
            return False
        selected = []
        advertisements = {
            (str(record.issuer), str(record.service_id)): record
            for record in self._select_advertisements()
        }
        for key, cached in self._peer_cache.items():
            record = advertisements.get(key)
            if record is not None:
                selected.append((record, cached.snapshot))
        self._last_result = self._result(self._combine(selected), ())
        return True


FederatedPeerCatalogSource = PeerCatalogSource


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_PEER_PRECEDENCE",
    "FederatedPeerCatalogSource",
    "MAX_PAGES_PER_PEER",
    "MAX_PEER_RECORDS",
    "MAX_PEERS",
    "PeerCatalogError",
    "PeerCatalogSource",
    "PeerCatalogTransport",
]
