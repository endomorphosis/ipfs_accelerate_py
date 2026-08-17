"""Hermetic tests for Libp2pDiscovery@1 and isolated AgntcyAdapter@1 (MCPP-060).

Acceptance:
* libp2p adapter has a hermetic test (publish / lookup over an in-memory mesh).
* AGNTCY either passes a live-optional test or is marked unsupported with a
  typed reject.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.registry.agntcy import (
    AGNTCY_ADAPTER_INTERFACE,
    AGNTCY_BLOCKER_ID,
    AGNTCY_LIVE_ENV,
    AGNTCY_PROVIDER_ID,
    AGNTCY_UNSUPPORTED_CODE,
    AgntcyAdapter,
    AgntcyUnsupportedError,
    create_agntcy_adapter,
    is_agntcy_supported,
    probe_agntcy_support,
    require_agntcy_supported,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    AGENT_ADVERTISEMENT_SCHEMA,
    REGISTRY_INTERFACE,
    Registry,
    RegistryNotFoundError,
    RegistryStaleError,
    is_execution_authority,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.registry.libp2p import (
    LIBP2P_DISCOVERY_INTERFACE,
    LIBP2P_DISCOVERY_PROTOCOL_ID,
    LIBP2P_PROVIDER_ID,
    InMemoryLibp2pMesh,
    Libp2pDiscovery,
    create_libp2p_discovery,
)

T0 = 1_700_000_000_000
IFACE_A = "bafkreigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
IFACE_B = "bafkreihxs3rpfcxfqeltptfyem7tjye7ro3v2jirue2vipi4un56agm2du"


def _ad(
    did: str,
    *,
    interfaces: Optional[list[str]] = None,
    ttl_ms: int = 60_000,
    published_at_ms: Optional[int] = T0,
    expires_at_ms: Optional[int] = None,
    health: Optional[str] = "healthy",
    utilization: Optional[int] = None,
    capacity: Optional[int] = None,
    skills: Optional[list[dict[str, Any]]] = None,
    policy_languages: Optional[list[str]] = None,
    proof_systems: Optional[list[str]] = None,
    signature: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": AGENT_ADVERTISEMENT_SCHEMA,
        "identity": {"did": did, "name": did.rsplit(":", 1)[-1]},
        "ttl_ms": ttl_ms,
        "interface_cids": list(interfaces if interfaces is not None else [IFACE_A]),
        "transports": ["libp2p", "mcp+p2p"],
    }
    if published_at_ms is not None:
        body["published_at_ms"] = published_at_ms
    if expires_at_ms is not None:
        body["expires_at_ms"] = expires_at_ms
    if health is not None:
        body["health"] = {"status": health}
    if utilization is not None or capacity is not None:
        load: Dict[str, Any] = {}
        if utilization is not None:
            load["utilization_millionths"] = utilization
        if capacity is not None:
            load["capacity_millionths"] = capacity
        body["load"] = load
    if skills is not None:
        body["skills"] = skills
    if policy_languages is not None:
        body["policy_languages"] = policy_languages
    if proof_systems is not None:
        body["proof_systems"] = proof_systems
    if signature is not None:
        body["signature"] = signature
    return body


class _Clock:
    def __init__(self, start: int = T0) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ms: int) -> None:
        self.now += delta_ms


def _solo(**kwargs: Any) -> Libp2pDiscovery:
    return create_libp2p_discovery(clock_ms=_Clock(), **kwargs)


class TestLibp2pDiscoveryContract:
    def test_interface_constants(self) -> None:
        assert LIBP2P_DISCOVERY_INTERFACE == "Libp2pDiscovery@1"
        assert LIBP2P_PROVIDER_ID == "libp2p"
        assert LIBP2P_DISCOVERY_PROTOCOL_ID.startswith("/mcp++/discovery/")

    def test_is_registry_and_not_execution_authority(self) -> None:
        reg = _solo()
        assert isinstance(reg, Registry)
        assert isinstance(reg, Libp2pDiscovery)
        assert reg.interface == LIBP2P_DISCOVERY_INTERFACE
        assert reg.family_interface == REGISTRY_INTERFACE
        assert reg.provider_id == LIBP2P_PROVIDER_ID
        assert is_execution_authority() is False
        ad = _ad("did:web:alpha")
        assert is_execution_authority(ad) is False
        reg.publish(ad)
        stats = reg.stats()
        assert stats["execution_authority"] is False
        assert stats["provider"] == "libp2p"
        assert stats["protocol_id"] == LIBP2P_DISCOVERY_PROTOCOL_ID
        reg.close()


class TestLibp2pHermeticPublishLookup:
    def test_publish_and_lookup_by_identity(self) -> None:
        reg = _solo()
        stored = reg.publish(_ad("did:web:alpha"))
        found = reg.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["identity"]["did"] == "did:web:alpha"
        assert IFACE_A in found["interface_cids"]
        assert "libp2p" in found["transports"]
        assert stored["identity"]["did"] == "did:web:alpha"
        reg.close()

    def test_refresh_withdraw_and_stale(self) -> None:
        clock = _Clock()
        reg = create_libp2p_discovery(clock_ms=clock)
        reg.publish(_ad("did:web:alpha", health="healthy"))
        refreshed = reg.refresh(_ad("did:web:alpha", health="degraded"))
        assert refreshed["health"]["status"] == "degraded"
        with pytest.raises(RegistryNotFoundError):
            reg.refresh(_ad("did:web:missing"))
        assert reg.withdraw("did:web:alpha") is True
        assert reg.lookup_by_identity("did:web:alpha") is None
        reg.publish(_ad("did:web:ttl", ttl_ms=1_000))
        clock.advance(5_000)
        assert reg.lookup_by_identity("did:web:ttl") is None
        assert reg.lookup_by_identity("did:web:ttl", include_stale=True) is not None
        reg.close()

    def test_publish_rejects_already_stale(self) -> None:
        reg = _solo()
        with pytest.raises(RegistryStaleError):
            reg.publish(
                _ad(
                    "did:web:stale",
                    published_at_ms=T0 - 120_000,
                    expires_at_ms=T0 - 60_000,
                    ttl_ms=1_000,
                )
            )
        assert reg.stats()["stale_rejects"] >= 1
        reg.close()

    def test_lookups_and_health_selection(self) -> None:
        reg = _solo()
        reg.publish(
            _ad(
                "did:web:git",
                skills=[{"id": "repo.status"}],
                policy_languages=["temporal-deontic@1"],
                proof_systems=["ucan"],
                health="healthy",
                utilization=100_000,
            )
        )
        reg.publish(
            _ad(
                "did:web:busy",
                health="degraded",
                utilization=900_000,
            )
        )
        assert reg.lookup_by_semantic_capability("repo.status")[0]["identity"]["did"] == "did:web:git"
        assert reg.lookup_by_policy("temporal-deontic@1")[0]["identity"]["did"] == "did:web:git"
        assert reg.lookup_by_proof("ucan")[0]["identity"]["did"] == "did:web:git"
        chosen = reg.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:git"
        reg.close()


class TestLibp2pMeshGossipHermetic:
    """Multi-peer publish/lookup over InMemoryLibp2pMesh — no real network."""

    def test_publish_on_peer_a_is_visible_on_peer_b(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        a = create_libp2p_discovery(transport=mesh, peer_id="peer-a", clock_ms=clock)
        b = create_libp2p_discovery(transport=mesh, peer_id="peer-b", clock_ms=clock)
        assert sorted(mesh.peer_ids()) == ["peer-a", "peer-b"]
        a.publish(_ad("did:web:alpha"))
        found = b.lookup_by_identity("did:web:alpha")
        assert found is not None
        assert found["identity"]["did"] == "did:web:alpha"
        assert IFACE_A in found["interface_cids"]
        assert b.stats()["remote_ingests"] >= 1
        assert a.stats()["broadcasts"] >= 1
        a.close()
        b.close()

    def test_withdraw_propagates_across_mesh(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        a = create_libp2p_discovery(transport=mesh, peer_id="peer-a", clock_ms=clock)
        b = create_libp2p_discovery(transport=mesh, peer_id="peer-b", clock_ms=clock)
        a.publish(_ad("did:web:alpha"))
        assert b.lookup_by_identity("did:web:alpha") is not None
        a.withdraw("did:web:alpha")
        assert b.lookup_by_identity("did:web:alpha") is None
        a.close()
        b.close()

    def test_refresh_propagates_health_change(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        a = create_libp2p_discovery(transport=mesh, peer_id="peer-a", clock_ms=clock)
        b = create_libp2p_discovery(transport=mesh, peer_id="peer-b", clock_ms=clock)
        a.publish(_ad("did:web:alpha", health="healthy"))
        a.refresh(_ad("did:web:alpha", health="degraded"))
        remote = b.lookup_by_identity("did:web:alpha")
        assert remote is not None
        assert remote["health"]["status"] == "degraded"
        chosen = b.select(interface_cid=IFACE_A)
        assert chosen is not None
        assert chosen["identity"]["did"] == "did:web:alpha"
        a.close()
        b.close()

    def test_three_peer_interface_lookup(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        peers = [
            create_libp2p_discovery(transport=mesh, peer_id=f"p{i}", clock_ms=clock)
            for i in range(3)
        ]
        peers[0].publish(_ad("did:web:a"))
        peers[1].publish(_ad("did:web:b", interfaces=[IFACE_B]))
        peers[2].publish(_ad("did:web:c"))
        found = peers[2].lookup_by_interface_cid(IFACE_A)
        dids = {ad["identity"]["did"] for ad in found}
        assert dids == {"did:web:a", "did:web:c"}
        for peer in peers:
            peer.close()

    def test_stale_remote_publish_is_rejected(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        a = create_libp2p_discovery(transport=mesh, peer_id="peer-a", clock_ms=clock)
        b = create_libp2p_discovery(transport=mesh, peer_id="peer-b", clock_ms=clock)
        with pytest.raises(RegistryStaleError):
            a.publish(
                _ad(
                    "did:web:stale",
                    published_at_ms=T0 - 120_000,
                    expires_at_ms=T0 - 60_000,
                    ttl_ms=1_000,
                )
            )
        assert b.lookup_by_identity("did:web:stale") is None
        a.close()
        b.close()

    def test_accept_remote_false_ignores_mesh(self) -> None:
        mesh = InMemoryLibp2pMesh()
        clock = _Clock()
        a = create_libp2p_discovery(transport=mesh, peer_id="peer-a", clock_ms=clock)
        b = create_libp2p_discovery(
            transport=mesh, peer_id="peer-b", clock_ms=clock, accept_remote=False
        )
        a.publish(_ad("did:web:alpha"))
        assert b.lookup_by_identity("did:web:alpha") is None
        assert b.stats()["accept_remote"] is False
        a.close()
        b.close()


class TestAgntcyUnsupportedTypedReject:
    def test_interface_and_default_unsupported(self) -> None:
        adapter = create_agntcy_adapter()
        status = probe_agntcy_support()
        assert status.supported is False
        assert is_agntcy_supported() is False
        assert isinstance(adapter, Registry)
        assert isinstance(adapter, AgntcyAdapter)
        assert adapter.interface == AGNTCY_ADAPTER_INTERFACE
        assert adapter.provider_id == AGNTCY_PROVIDER_ID
        assert adapter.family_interface == REGISTRY_INTERFACE
        assert adapter.supported is False
        stats = adapter.stats()
        assert stats["execution_authority"] is False
        assert stats["blocker_id"] == AGNTCY_BLOCKER_ID
        assert stats["support"]["supported"] is False

    def test_publish_raises_typed_reject(self) -> None:
        adapter = create_agntcy_adapter()
        with pytest.raises(AgntcyUnsupportedError) as err:
            adapter.publish(_ad("did:web:alpha"))
        assert err.value.code == AGNTCY_UNSUPPORTED_CODE
        assert err.value.blocker_id == AGNTCY_BLOCKER_ID
        payload = err.value.to_dict()
        assert payload["supported"] is False
        assert payload["provider"] == AGNTCY_PROVIDER_ID
        assert adapter.stats()["reject_count"] >= 1

    def test_all_registry_ops_typed_reject(self) -> None:
        adapter = create_agntcy_adapter()
        ops = [
            lambda: adapter.refresh(_ad("did:web:x")),
            lambda: adapter.withdraw("did:web:x"),
            lambda: adapter.lookup_by_identity("did:web:x"),
            lambda: adapter.lookup_by_interface_cid(IFACE_A),
            lambda: adapter.lookup_by_semantic_capability("git"),
            lambda: adapter.lookup_by_policy("temporal-deontic@1"),
            lambda: adapter.lookup_by_proof("ucan"),
            lambda: adapter.select(),
            lambda: adapter.list_all(),
        ]
        for op in ops:
            with pytest.raises(AgntcyUnsupportedError) as ei:
                op()
            assert ei.value.code == AGNTCY_UNSUPPORTED_CODE
        assert adapter.stats()["reject_count"] == len(ops)

    def test_require_agntcy_supported_raises_typed_reject(self) -> None:
        with pytest.raises(AgntcyUnsupportedError) as ei:
            require_agntcy_supported()
        assert ei.value.code == AGNTCY_UNSUPPORTED_CODE
        assert ei.value.blocker_id == AGNTCY_BLOCKER_ID

    def test_live_optional_probe_with_simulated_sdk(self) -> None:
        """Live-optional path: when SDK is present and env flag set, probe passes.

        The adapter still typed-rejects operations because the reviewed live
        binding is incomplete — we never invent discovery results.
        """
        env = {AGNTCY_LIVE_ENV: "1"}
        status = probe_agntcy_support(
            env=env, force_sdk_available=True, force_sdk_module="agntcy"
        )
        assert status.supported is True
        assert is_agntcy_supported(env=env, force_sdk_available=True) is True
        adapter = create_agntcy_adapter(
            allow_live=True,
            env=env,
            force_sdk_available=True,
            force_sdk_module="agntcy",
        )
        assert adapter.support_status.supported is True
        assert adapter.supported is False
        with pytest.raises(AgntcyUnsupportedError) as ei:
            adapter.publish(_ad("did:web:alpha"))
        assert ei.value.code == AGNTCY_UNSUPPORTED_CODE
        assert ei.value.blocker_id == "agntcy-live-binding-incomplete"

    def test_sdk_present_but_live_flag_off_is_unsupported(self) -> None:
        status = probe_agntcy_support(force_sdk_available=True, force_sdk_module="agntcy")
        assert status.supported is False
        assert status.blocker_id == "agntcy-live-not-enabled"

    def test_default_runtime_without_sdk_documents_blocker(self) -> None:
        status = probe_agntcy_support()
        assert status.supported is False
        assert status.blocker_id == AGNTCY_BLOCKER_ID
        assert status.blocker_summary
        assert "AGNTCY" in status.blocker_summary
        adapter = create_agntcy_adapter()
        with pytest.raises(AgntcyUnsupportedError) as ei:
            adapter.publish(_ad("did:web:alpha"))
        assert ei.value.code == AGNTCY_UNSUPPORTED_CODE
