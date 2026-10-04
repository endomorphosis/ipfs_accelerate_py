"""Synthetic, non-inference tests for the Leanstral MCP++ topology contract."""

from dataclasses import asdict, replace
import json

import pytest

from ipfs_accelerate_py.mcplusplus_module.leanstral_topology import (
    CapabilityClaim,
    IndependentDialObservation,
    InterfaceAddress,
    LEANSTRAL_P2P_LISTEN_ADDR,
    LEANSTRAL_P2P_PORT,
    LeanstralTopologyObservation,
    ProbeObservation,
    default_capability_claims,
    leanstral_advertised_multiaddrs,
    normalize_served_model_record,
    select_advertised_ipv4,
    validate_leanstral_topology,
    validate_leanstral_topology_mapping,
)
from ipfs_accelerate_py.mcplusplus_module.p2p_transport import (
    DEFAULT_BOOTSTRAP_PEERS,
    MCPp2pNode,
    _bootstrap_dial_candidates,
)
from ipfs_accelerate_py.mcplusplus_module.trio.server import (
    ServerConfig,
    _build_p2p_service_metadata,
)


SERVER_PEER_ID = "12D3KooWLeanstralServerExactPeer"
CLIENT_PEER_ID = "12D3KooWIndependentDialerPeer"


def _valid_observation() -> LeanstralTopologyObservation:
    interfaces = (
        InterfaceAddress("wlP9s9", "172.30.4.2", is_up=True, scope="lan"),
        InterfaceAddress("tun0", "10.8.0.99", is_up=True, scope="lan"),
        InterfaceAddress("tun1", "10.10.0.14", is_up=True, scope="lan"),
        InterfaceAddress("docker0", "172.17.0.1", is_up=False, scope="container"),
        InterfaceAddress("lo", "127.0.0.1", is_up=True, scope="lan"),
        InterfaceAddress("eth9", "192.168.50.10", is_up=True, scope="unrelated"),
    )
    allowlist = ("wlP9s9", "tun0", "tun1")
    selected = select_advertised_ipv4(
        interfaces,
        allowed_interfaces=allowlist,
    ).selected
    multiaddrs = leanstral_advertised_multiaddrs(selected, peer_id=SERVER_PEER_ID)
    capabilities = default_capability_claims(rendezvous_implemented=True)
    model = normalize_served_model_record(
        transport_model_id="Frosty40/Leanstral-1.5-119B-A6B-NVFP4",
        endpoint="http://127.0.0.1:8080/v1",
        owned_by="llama.cpp",
        metadata={"n_ctx": 8192},
    )
    return LeanstralTopologyObservation(
        p2p_requested=True,
        p2p_enabled=True,
        listen_addrs=(LEANSTRAL_P2P_LISTEN_ADDR,),
        peer_id=SERVER_PEER_ID,
        advertised_multiaddrs=multiaddrs,
        interfaces=interfaces,
        advertise_interface_allowlist=allowlist,
        bootstrap_exercises=(
            ProbeObservation(
                mechanism="bootstrap",
                target="/dnsaddr/bootstrap.libp2p.io/p2p/QmBootstrap",
                attempted=True,
                success=True,
                timeout_s=5.0,
                duration_ms=120.0,
            ),
        ),
        rendezvous_exercises=(
            ProbeObservation(
                mechanism="rendezvous",
                target=SERVER_PEER_ID,
                attempted=True,
                success=True,
                timeout_s=5.0,
                duration_ms=85.0,
                observer_peer_id=CLIENT_PEER_ID,
                namespace="leanstral-local",
            ),
        ),
        capabilities=capabilities,
        independent_dial=IndependentDialObservation(
            dialer_peer_id=CLIENT_PEER_ID,
            target_peer_id=SERVER_PEER_ID,
            target_multiaddr=multiaddrs[1],
            attempted=True,
            success=True,
            timeout_s=5.0,
            duration_ms=42.0,
        ),
        served_models=(model,),
        server_instance_count=1,
        inference_attempted=False,
        # HTTP may remain on 8000; it is explicitly not the P2P port.
        http_port=8000,
    )


def test_address_policy_selects_all_permitted_active_local_addresses():
    observation = _valid_observation()

    selection = select_advertised_ipv4(
        observation.interfaces,
        allowed_interfaces=observation.advertise_interface_allowlist,
    )

    assert selection.selected == ("10.10.0.14", "10.8.0.99", "172.30.4.2")
    assert "172.17.0.1" not in selection.selected
    assert "127.0.0.1" not in selection.selected
    assert "192.168.50.10" not in selection.selected


def test_complete_topology_receipt_is_valid_cidv1_and_inference_free():
    from multiformats import CID

    result = validate_leanstral_topology(_valid_observation())

    assert result.valid is True
    assert result.errors == ()
    decoded = CID.decode(result.receipt_cid)
    assert decoded.version == 1
    assert decoded.codec.name == "raw"
    assert decoded.hashfun.name == "sha2-256"
    assert result.receipt["receipt_cid"] == result.receipt_cid
    assert result.receipt["contract"] == {
        "logical_model_id": "leanstral_local",
        "http_transport": "llamacpp",
        "p2p_protocol": "/mcp+p2p/1.0.0",
        "p2p_port": 19001,
        "listen_addr": "/ip4/0.0.0.0/tcp/19001",
        "requires_fresh_external_receipt": True,
        "inference_required": False,
    }
    assert len(result.receipt["observation"]["advertised_multiaddrs"]) == 3
    assert result.receipt["observation"]["inference_attempted"] is False


def test_json_compatible_topology_mapping_uses_the_same_strict_contract():
    evidence = json.loads(json.dumps(asdict(_valid_observation())))

    result = validate_leanstral_topology_mapping(evidence)

    assert result.valid is True
    assert result.receipt["observation"] == evidence


def test_json_decoder_rejects_coercion_and_nested_field_tampering():
    valid = json.loads(json.dumps(asdict(_valid_observation())))
    mutations = (
        ("server_instance_count", True, "server_instance_count must be an integer"),
        ("peer_id", 123, "peer_id must be a string"),
        ("http_port", "8000", "http_port must be an integer"),
    )
    for field_name, bad_value, message in mutations:
        evidence = json.loads(json.dumps(valid))
        evidence[field_name] = bad_value
        with pytest.raises(ValueError, match=message):
            validate_leanstral_topology_mapping(evidence)

    capability_coercion = json.loads(json.dumps(valid))
    capability_coercion["capabilities"]["mcp_stream"]["advertised"] = 1
    with pytest.raises(ValueError, match="advertised must be boolean"):
        validate_leanstral_topology_mapping(capability_coercion)

    nested_injection = json.loads(json.dumps(valid))
    nested_injection["independent_dial"]["coerced_success"] = True
    with pytest.raises(ValueError, match="independent_dial fields differ"):
        validate_leanstral_topology_mapping(nested_injection)


def test_contract_rejects_wrong_port_duplicates_overclaims_and_inference():
    valid = _valid_observation()
    bad_model = dict(valid.served_models[0])
    bad_model["id"] = "raw-transport-filename"
    bad_capabilities = dict(valid.capabilities)
    bad_capabilities["pubsub"] = CapabilityClaim(
        configured=True,
        implemented=False,
        advertised=True,
        policy="invented",
    )
    bad_dial = replace(
        valid.independent_dial,
        dialer_peer_id=SERVER_PEER_ID,
        target_multiaddr=f"/ip4/172.30.4.2/tcp/8000/p2p/{SERVER_PEER_ID}",
        success=False,
    )
    observation = replace(
        valid,
        listen_addrs=("/ip4/0.0.0.0/tcp/8000",),
        advertised_multiaddrs=(f"/ip4/172.30.4.2/tcp/8000/p2p/{SERVER_PEER_ID}",),
        capabilities=bad_capabilities,
        independent_dial=bad_dial,
        served_models=(bad_model, valid.served_models[0]),
        server_instance_count=2,
        inference_attempted=True,
    )

    result = validate_leanstral_topology(observation)

    assert result.valid is False
    assert "listen_addr_not_exact_wildcard_19001" in result.errors
    assert "advertised_multiaddr_0_wrong_port" in result.errors
    assert "advertised_multiaddr_0_uses_http_default_port" in result.errors
    assert "capability_pubsub_advertisement_mismatch" in result.errors
    assert "capability_pubsub_overclaimed" in result.errors
    assert "independent_dial_not_independent" in result.errors
    assert "served_model_record_count_not_one" in result.errors
    assert "server_instance_count_not_one" in result.errors
    assert "inference_was_attempted" in result.errors


def test_container_only_or_stale_addresses_fail_closed():
    valid = _valid_observation()
    interfaces = (
        InterfaceAddress("lo", "127.0.0.1", is_up=True, scope="lan"),
        InterfaceAddress("docker0", "172.17.0.1", is_up=True, scope="container"),
        InterfaceAddress("eth0", "172.18.0.2", is_up=False, scope="lan"),
        InterfaceAddress("eth9", "192.168.1.5", is_up=True, scope="unrelated"),
    )
    observation = replace(
        valid,
        interfaces=interfaces,
        advertise_interface_allowlist=("lo", "docker0", "eth0", "eth9"),
        advertised_multiaddrs=(),
        independent_dial=replace(valid.independent_dial, target_multiaddr="", success=False),
    )

    result = validate_leanstral_topology(observation)

    assert result.valid is False
    assert "policy_selected_advertised_addresses_empty" in result.errors
    assert "independent_dial_target_not_advertised" in result.errors


def test_p2p_node_defaults_to_custom_port_and_truthful_capabilities(monkeypatch):
    for name in (
        "MCPPP_P2P_LISTEN_ADDRS",
        "MCPPP_P2P_BOOTSTRAP_PEERS",
        "MCPPP_P2P_ADVERTISE_ADDRS",
        "MCPPP_P2P_ADVERTISE_INTERFACES",
    ):
        monkeypatch.delenv(name, raising=False)

    node = MCPp2pNode()
    status = node.to_dict()

    assert status["listen_addrs"] == [LEANSTRAL_P2P_LISTEN_ADDR]
    assert status["listen_port"] == LEANSTRAL_P2P_PORT
    assert status["listen_port"] != 8000
    assert status["bootstrap"]["configured_peers"] == DEFAULT_BOOTSTRAP_PEERS
    assert len(status["bootstrap"]["configured_peers"]) > 0
    assert status["capabilities"]["pubsub"]["implemented"] is False
    assert status["capabilities"]["pubsub"]["advertised"] is False
    assert status["capabilities"]["floodsub"]["implemented"] is False
    assert status["capabilities"]["floodsub"]["advertised"] is False


def test_bootstrap_attempt_receipt_matches_strict_probe_shape(monkeypatch):
    import trio

    node = MCPp2pNode(bootstrap_peers=[])

    async def connected(_target):
        return True

    monkeypatch.setattr(node, "_connect_bootstrap", connected)
    trio.run(
        node._connect_bootstrap_with_timeout,
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmBootstrap",
    )

    receipt = node.to_dict()["bootstrap"]["attempts"][0]
    assert set(receipt) == {
        "mechanism",
        "target",
        "attempted",
        "success",
        "timeout_s",
        "duration_ms",
        "error",
        "observer_peer_id",
        "namespace",
        "details",
    }
    assert receipt["success"] is True
    assert receipt["namespace"] == ""
    assert receipt["details"] == {}


def test_dnsaddr_bootstrap_resolves_to_same_peer_plain_tcp(monkeypatch):
    import trio
    from multiaddr import Multiaddr

    peer_id = DEFAULT_BOOTSTRAP_PEERS[0].rsplit("/p2p/", 1)[-1]

    async def resolved(_value):
        return [
            Multiaddr(f"/dns/bootstrap.example/tcp/443/wss/p2p/{peer_id}"),
            Multiaddr(f"/dns/bootstrap.example/tcp/4001/p2p/{peer_id}"),
            Multiaddr(
                "/dns/bootstrap.example/tcp/4001/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa"
            ),
        ]

    monkeypatch.setattr(Multiaddr, "resolve", resolved)
    candidates = trio.run(
        _bootstrap_dial_candidates,
        DEFAULT_BOOTSTRAP_PEERS[0],
    )

    assert candidates == (f"/dns/bootstrap.example/tcp/4001/p2p/{peer_id}",)


def test_rendezvous_exercise_uses_exact_peer_and_is_bounded(monkeypatch):
    import trio
    from ipfs_accelerate_py.mcplusplus_module.p2p import connectivity

    configured = {}

    class _Connectivity:
        def __init__(self, config):
            configured["config"] = config
            self.implemented = {"rendezvous": False}

        async def configure_rendezvous(self, host, rendezvous_peer=None):
            configured["host"] = host
            configured["peer"] = rendezvous_peer
            self.implemented["rendezvous"] = True

        async def rendezvous_register(self, namespace):
            configured["registered_namespace"] = namespace
            return True

        async def rendezvous_discover(self, namespace):
            configured["discovered_namespace"] = namespace
            return ["/ip4/10.8.0.99/tcp/19001/p2p/12D3KooWPeer"]

    monkeypatch.setattr(connectivity, "UniversalConnectivity", _Connectivity)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER",
        "12D3KooWExactRendezvousPeer",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_P2P_RENDEZVOUS_NS", "leanstral-local")
    node = MCPp2pNode(bootstrap_peers=[])

    class _Host:
        @staticmethod
        def get_id():
            return SERVER_PEER_ID

        @staticmethod
        def get_addrs():
            return []

    node._host = _Host()

    async def _exercise():
        return await node.exercise_rendezvous(timeout=2.0)

    receipt = trio.run(_exercise)

    assert configured["host"] is node._host
    assert configured["peer"] == "12D3KooWExactRendezvousPeer"
    assert configured["registered_namespace"] == "leanstral-local"
    assert receipt["attempted"] is True
    assert receipt["success"] is True
    assert receipt["timeout_s"] == 2.0
    assert receipt["duration_ms"] <= 2250.0
    assert node.to_dict()["capabilities"]["rendezvous"] == {
        "configured": True,
        "implemented": True,
        "advertised": False,
    }
    assert node.to_dict()["rendezvous"]["client"]["exercise_success"] is True


def test_rendezvous_factory_passes_peer_required_by_current_libp2p(monkeypatch):
    from ipfs_accelerate_py.mcplusplus_module.p2p import libp2p_runtime

    calls = []
    expected = object()

    def _make(symbols, *args, **kwargs):
        calls.append((symbols, args, kwargs))
        return expected

    monkeypatch.setattr(libp2p_runtime, "_make_first_available", _make)
    host = object()
    peer = object()

    client = libp2p_runtime.make_rendezvous_client(host, peer)

    assert client is expected
    assert calls[0][1] == (host, peer)


def test_rendezvous_service_mounts_on_the_service_peer(monkeypatch):
    import ipfs_accelerate_py.mcplusplus_module.p2p_transport as transport

    mounted = object()
    captured = {}

    def _mount(host):
        captured["host"] = host
        return mounted

    monkeypatch.setenv(
        "MCPPP_P2P_RENDEZVOUS_SERVICE",
        "same_as_service_peer",
    )
    monkeypatch.setattr(transport, "make_rendezvous_service", _mount)
    node = MCPp2pNode(bootstrap_peers=[])

    class _Host:
        @staticmethod
        def get_id():
            return SERVER_PEER_ID

        @staticmethod
        def get_addrs():
            return []

    node._host = _Host()
    node._operational = True

    assert node._mount_rendezvous_service() is True
    assert captured["host"] is node._host
    status = node.to_dict()
    assert status["rendezvous"]["service"] == {
        "mode": "same_as_service_peer",
        "configured": True,
        "implemented": True,
        "peer_id": SERVER_PEER_ID,
    }
    assert status["rendezvous"]["external_exercise_receipt_required"] is True
    assert status["capabilities"]["rendezvous"]["advertised"] is True


def test_rendezvous_self_client_cannot_count_as_independent(monkeypatch):
    import trio

    monkeypatch.setenv(
        "IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER",
        SERVER_PEER_ID,
    )
    node = MCPp2pNode(bootstrap_peers=[])

    class _Host:
        @staticmethod
        def get_id():
            return SERVER_PEER_ID

        @staticmethod
        def get_addrs():
            return []

    node._host = _Host()

    async def _exercise():
        return await node.exercise_rendezvous(timeout=1.0)

    receipt = trio.run(_exercise)

    assert receipt["attempted"] is False
    assert receipt["success"] is False
    assert receipt["error"] == "rendezvous_client_not_independent"


def test_service_metadata_never_aliases_http_port_as_p2p_port():
    class _Node:
        listen_port = LEANSTRAL_P2P_PORT

        @staticmethod
        def to_dict():
            return {
                "protocol": "/mcp+p2p/1.0.0",
                "capabilities": default_capability_claims(),
            }

    model = normalize_served_model_record(
        transport_model_id="example/Leanstral",
        endpoint="http://127.0.0.1:8080/v1",
    )
    metadata = _build_p2p_service_metadata(
        config=ServerConfig(port=8000),
        node=_Node(),
        served_models=[model],
    )

    assert "port" not in metadata
    assert metadata["http_port"] == 8000
    assert metadata["p2p_port"] == 19001
    assert metadata["models"] == ["leanstral_local"]
    assert metadata["served_models"][0]["transport"] == "llamacpp"
    assert metadata["node_ownership"] == "process_singleton"
