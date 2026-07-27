from __future__ import annotations

import copy
import time

import pytest
import trio

from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog, RefreshPolicy
from ipfs_accelerate_py.model_catalog.schema import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    Modality,
    ModelDescriptor,
    Operation,
    ProviderDescriptor,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.snapshot import paginate_snapshot
from ipfs_accelerate_py.model_catalog.sources.peers import PeerCatalogSource
from ipfs_accelerate_py.model_catalog.sources.static import StaticCatalogSource
from ipfs_accelerate_py.mcplusplus_module.service_registry import (
    CATALOG_ENDPOINT_PROTOCOL,
    ServiceRecord,
    ServiceRegistry,
)
from ipfs_accelerate_py.mcplusplus_module.trio.server import (
    ServerConfig,
    _build_catalog_service_record,
)


def _snapshot(provider_name: str, model_names=("one", "two")) -> CatalogSnapshot:
    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_GENERATE,),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
    )
    provider = ProviderDescriptor(
        name=provider_name,
        capabilities=(capability,),
    )
    models = tuple(
        ModelDescriptor(
            provider_id=provider.provider_id,
            name=name,
            capabilities=(capability,),
        )
        for name in model_names
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=models[0].model_id,
        name="primary",
        endpoint_uri="https://peer.example.test/v1",
        capabilities=(capability,),
    )
    binding = RouterBinding(
        router="llm_router",
        provider_id=provider.provider_id,
        model_id=models[0].model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_GENERATE,),
    )
    return CatalogSnapshot(
        providers=(provider,),
        models=models,
        deployments=(deployment,),
        bindings=(binding,),
    )


def _advertisement(
    peer_id: str,
    snapshot: CatalogSnapshot,
    *,
    issued_at: float | None = None,
    ttl: float = 300.0,
) -> ServiceRecord:
    issued = time.time() if issued_at is None else issued_at
    record = ServiceRecord(
        service_name="ipfs-accelerate-mcp",
        peer_id=peer_id,
        issuer=peer_id,
        multiaddrs=[f"/memory/{peer_id}"],
        catalog_cid=snapshot.revision,
        catalog_revision=snapshot.revision,
        operation_summary=["text.generate"],
        interface_cids=["cidv1-ai-catalog"],
        endpoint_protocol=CATALOG_ENDPOINT_PROTOCOL,
        issued_at=issued,
        expires_at=issued + ttl,
        metadata={"server": peer_id},
    )
    record.sign()
    return record


class OfflineTransport:
    """Deterministic authorized transport with no network primitives."""

    authorized = True

    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []
        self.fail = set()
        self.mismatch = set()

    def fetch_catalog_page(
        self, advertisement, *, record_type, cursor, limit
    ):
        self.calls.append(
            (
                advertisement.peer_id,
                advertisement.catalog_revision,
                record_type,
                cursor,
                limit,
            )
        )
        if advertisement.peer_id in self.fail:
            raise ConnectionError("offline peer")
        page = paginate_snapshot(
            self.snapshots[advertisement.peer_id],
            record_type,
            limit=limit,
            cursor=cursor,
        ).to_dict()
        if (
            advertisement.peer_id in self.mismatch
            and record_type == "providers"
            and page["items"]
        ):
            # Keep the advertised page revision intact while changing canonical
            # content, exercising the final reconstructed-snapshot CID check.
            page["items"][0]["description"] = "tampered"
        return copy.deepcopy(page)


class FakeNode:
    def __init__(self):
        self._peers = {"remote": object()}
        self.peer_id = "local-peer"
        self.multiaddrs = ["/memory/local-peer"]
        self.listen_port = 4001
        self.sent = []

    def to_dict(self):
        return {
            "protocol": CATALOG_ENDPOINT_PROTOCOL,
            "capabilities": {},
        }

    async def call_tool(self, peer_id, method, params, timeout):
        self.sent.append((peer_id, method, copy.deepcopy(params), timeout))
        return {"status": "accepted"}


def test_compact_advertisement_has_catalog_identity_and_no_model_inventory():
    snapshot = _snapshot("local")
    record = _build_catalog_service_record(
        config=ServerConfig(name="test", port=8080),
        node=FakeNode(),
        snapshot=snapshot,
        now=100.0,
    )
    payload = record.to_dict()

    assert payload["issuer"] == "local-peer"
    assert payload["service_id"].startswith("service_")
    assert payload["catalog_cid"] == snapshot.revision
    assert payload["catalog_revision"] == snapshot.revision
    assert payload["operation_summary"]
    assert len(payload["interface_cids"]) == 1
    assert payload["endpoint_protocol"] == CATALOG_ENDPOINT_PROTOCOL
    assert payload["issued_at"] == 100.0
    assert payload["expires_at"] == 400.0
    assert record.verify_signature()
    assert "models" not in payload["metadata"]
    assert "served_models" not in payload["metadata"]

    tampered = ServiceRecord.from_dict(payload)
    tampered.catalog_revision = "tampered"
    assert not tampered.verify_signature()


def test_dynamic_catalog_revision_is_broadcast_without_server_restart():
    first = _snapshot("first", ("one",))
    second = _snapshot("second", ("two",))
    current = [first]
    registry = ServiceRegistry()
    node = FakeNode()
    record = _advertisement(node.peer_id, first)
    events = []
    registry.on_change(
        "test", lambda event, changed: events.append((event, changed.catalog_revision))
    )
    registry.register_local(record, catalog_provider=lambda: current[0])

    async def exercise():
        await registry.advertise_once(node, now=1000.0)
        current[0] = second
        await registry.advertise_once(node, now=1001.0)

    trio.run(exercise)

    announced = [
        call[2]["record"]
        for call in node.sent
        if call[1] == "_mcppp_service_announce"
    ]
    assert [item["catalog_revision"] for item in announced] == [
        first.revision,
        second.revision,
    ]
    assert any(event == ("catalog_update", second.revision) for event in events)
    assert registry.get_local(record.service_name).catalog_revision == second.revision


def test_peer_source_fetches_all_pages_only_through_authorized_transport():
    snapshot = _snapshot("remote", ("a", "b", "c"))
    record = _advertisement("peer-a", snapshot)
    transport = OfflineTransport({"peer-a": snapshot})
    source = PeerCatalogSource(
        [record],
        transport,
        trust_domain="tenant-a",
        page_size=1,
    )

    assert source.load().snapshot.providers == ()
    assert transport.calls == []

    result = source.refresh()
    fetched = result.snapshot
    assert len(fetched.providers) == 1
    assert len(fetched.models) == 3
    assert len(fetched.deployments) == 1
    assert len(fetched.bindings) == 1
    assert len(transport.calls) == 6
    assert all(call[4] == 1 for call in transport.calls)
    provenance = fetched.providers[0].provenance[0]
    assert provenance.source == source.source
    assert provenance.issuer == "tenant-a:peer-a"
    assert fetched.providers[0].provider_id != snapshot.providers[0].provider_id


def test_peer_source_rejects_transport_without_explicit_authorization():
    snapshot = _snapshot("remote", ("a",))
    record = _advertisement("peer-a", snapshot)
    transport = OfflineTransport({"peer-a": snapshot})
    transport.authorized = False

    result = PeerCatalogSource(
        [record], transport, trust_domain="tenant-a"
    ).refresh()

    assert result.snapshot.providers == ()
    assert [item.code for item in result.diagnostics] == ["peer_fetch_failed"]
    assert transport.calls == []


def test_partial_and_duplicate_peers_are_isolated_and_deduplicated():
    good = _snapshot("good", ("a",))
    unavailable = _snapshot("unavailable", ("b",))
    good_record = _advertisement("peer-good", good)
    duplicate = ServiceRecord.from_dict(good_record.to_dict())
    bad_record = _advertisement("peer-bad", unavailable)
    transport = OfflineTransport(
        {"peer-good": good, "peer-bad": unavailable}
    )
    transport.fail.add("peer-bad")

    result = PeerCatalogSource(
        [duplicate, bad_record, good_record],
        transport,
        trust_domain="tenant-a",
    ).refresh()

    assert len(result.snapshot.providers) == 1
    assert [item.code for item in result.diagnostics] == ["peer_fetch_failed"]
    assert sum(call[0] == "peer-good" for call in transport.calls) == 4
    assert sum(call[0] == "peer-bad" for call in transport.calls) == 1


def test_cid_mismatch_and_stale_advertisements_fail_closed():
    snapshot = _snapshot("remote", ("a",))
    mismatch = _advertisement("peer-mismatch", snapshot)
    expired = _advertisement(
        "peer-expired",
        snapshot,
        issued_at=time.time() - 20.0,
        ttl=1.0,
    )
    transport = OfflineTransport(
        {"peer-mismatch": snapshot, "peer-expired": snapshot}
    )
    transport.mismatch.add("peer-mismatch")

    result = PeerCatalogSource(
        [expired, mismatch],
        transport,
        trust_domain="tenant-a",
    ).refresh()

    assert result.snapshot.providers == ()
    assert [item.code for item in result.diagnostics] == ["peer_fetch_failed"]
    assert all(call[0] != "peer-expired" for call in transport.calls)


def test_disconnect_removes_peer_records_on_next_generation():
    snapshot = _snapshot("remote", ("a",))
    record = _advertisement("peer-a", snapshot)
    registry = ServiceRegistry()
    assert registry.add_remote(record)
    transport = OfflineTransport({"peer-a": snapshot})
    source = PeerCatalogSource(
        registry,
        transport,
        trust_domain="tenant-a",
    )

    assert len(source.refresh().snapshot.providers) == 1
    assert registry.remove_remote("peer-a") == 1
    assert source.refresh().snapshot.providers == ()
    assert source.load().snapshot.providers == ()


def test_restart_and_trust_domains_have_deterministic_isolated_identities():
    snapshot = _snapshot("remote", ("a",))
    record = _advertisement("peer-a", snapshot)

    first = PeerCatalogSource(
        [record],
        OfflineTransport({"peer-a": snapshot}),
        trust_domain="tenant-a",
    ).refresh().snapshot
    restarted = PeerCatalogSource(
        [record],
        OfflineTransport({"peer-a": snapshot}),
        trust_domain="tenant-a",
    ).refresh().snapshot
    other_domain = PeerCatalogSource(
        [record],
        OfflineTransport({"peer-a": snapshot}),
        trust_domain="tenant-b",
    ).refresh().snapshot

    assert restarted.revision == first.revision
    assert restarted.providers[0].provider_id == first.providers[0].provider_id
    assert other_domain.providers[0].provider_id != first.providers[0].provider_id


def test_peer_records_cannot_override_trusted_local_records():
    remote = _snapshot("shared", ("remote-model",))
    record = _advertisement("peer-a", remote)
    peer_source = PeerCatalogSource(
        [record],
        OfflineTransport({"peer-a": remote}),
        trust_domain="tenant-a",
    )
    peer_source.refresh()
    local_source = StaticCatalogSource(
        [
            {
                "provider": "shared",
                "model": "local-model",
                "operations": ["text.generate"],
            }
        ],
        source="local.static",
        precedence=10,
    )
    catalog = AIServiceCatalog()
    catalog.register_source("local.static", local_source)
    catalog.register_source(
        peer_source.source,
        peer_source,
        side_effecting=True,
    )

    providers = catalog.snapshot().providers
    local = next(item for item in providers if item.name == "shared")
    assert local.provider_id == ProviderDescriptor(name="shared").provider_id
    assert len(providers) == 2

    refreshed = catalog.refresh(
        [peer_source.source],
        policy=RefreshPolicy(
            allow_side_effects=True,
            allowed_sources=(peer_source.source,),
        ),
    )
    assert not refreshed.failed


def test_service_registry_pages_are_revision_bound_and_restart_safe():
    snapshot = _snapshot("local", ("a", "b"))
    record = _advertisement("local-peer", snapshot)
    registry = ServiceRegistry()
    registry.register_local(record, catalog_provider=lambda: snapshot)
    page = registry.handle_catalog_page(
        {
            "service_name": record.service_name,
            "catalog_revision": snapshot.revision,
            "record_type": "models",
            "limit": 1,
            "cursor": None,
        }
    )
    assert page["snapshot_revision"] == snapshot.revision
    assert page["next_cursor"]

    with pytest.raises(ValueError, match="stale"):
        registry.handle_catalog_page(
            {
                "service_name": record.service_name,
                "catalog_revision": "old-revision",
                "record_type": "models",
                "limit": 1,
                "cursor": None,
            }
        )
    assert registry.unregister_local(record.service_name, peer_id="local-peer")
    assert registry.get_local(record.service_name) is None
