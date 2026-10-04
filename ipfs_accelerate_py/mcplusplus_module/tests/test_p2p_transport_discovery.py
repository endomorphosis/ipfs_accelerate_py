"""Socket-free regression tests for MCP++ mDNS peer discovery."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import trio

from ipfs_accelerate_py.mcplusplus_module.p2p_transport import MCPp2pNode


SELF_PEER_ID = "12D3KooWLocalPeer"
REMOTE_PEER_ID = "12D3KooWRemotePeer"


def _install_fake_zeroconf(monkeypatch, *, advertised_names=()):
    state = SimpleNamespace(zeroconfs=[], browsers=[])

    class FakeInfo:
        port = 19001

        @staticmethod
        def parsed_addresses():
            return ["10.8.0.99"]

    class FakeZeroconf:
        def __init__(self):
            self.close_calls = 0
            state.zeroconfs.append(self)

        @staticmethod
        def get_service_info(_service_type, _name):
            return FakeInfo()

        def close(self):
            self.close_calls += 1

    class FakeServiceBrowser:
        def __init__(self, zeroconf, service_type, listener):
            self.cancel_calls = 0
            state.browsers.append(self)
            for name in advertised_names:
                listener.add_service(zeroconf, service_type, name)

        def cancel(self):
            self.cancel_calls += 1

    module = ModuleType("zeroconf")
    module.Zeroconf = FakeZeroconf
    module.ServiceBrowser = FakeServiceBrowser
    monkeypatch.setitem(sys.modules, "zeroconf", module)
    return state


def _node_with_identity(peer_id=SELF_PEER_ID):
    node = MCPp2pNode(bootstrap_peers=[])
    node._host = SimpleNamespace(
        get_id=lambda: peer_id,
        get_addrs=lambda: [],
    )
    return node


def test_mdns_disabled_does_not_construct_zeroconf(monkeypatch):
    state = _install_fake_zeroconf(monkeypatch)
    monkeypatch.setenv("MCPPP_P2P_MDNS", "0")

    result = trio.run(_node_with_identity().discover_peers)

    assert result == []
    assert state.zeroconfs == []
    assert state.browsers == []


def test_discovery_normal_completion_cleans_up_browser_and_zeroconf(monkeypatch):
    state = _install_fake_zeroconf(monkeypatch)
    monkeypatch.setenv("MCPPP_P2P_MDNS", "1")

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(trio, "sleep", no_sleep)
    result = trio.run(_node_with_identity().discover_peers)

    assert result == []
    assert len(state.browsers) == 1
    assert state.browsers[0].cancel_calls == 1
    assert len(state.zeroconfs) == 1
    assert state.zeroconfs[0].close_calls == 1


def test_discovery_error_cleans_up_browser_and_zeroconf(monkeypatch):
    state = _install_fake_zeroconf(monkeypatch)
    monkeypatch.setenv("MCPPP_P2P_MDNS", "1")

    async def fail_sleep(_delay):
        raise RuntimeError("synthetic discovery failure")

    monkeypatch.setattr(trio, "sleep", fail_sleep)
    result = trio.run(_node_with_identity().discover_peers)

    assert result == []
    assert state.browsers[0].cancel_calls == 1
    assert state.zeroconfs[0].close_calls == 1


def test_discovery_cancellation_cleans_up_browser_and_zeroconf(monkeypatch):
    state = _install_fake_zeroconf(monkeypatch)
    monkeypatch.setenv("MCPPP_P2P_MDNS", "1")
    node = _node_with_identity()

    async def cancel_discovery():
        with trio.move_on_after(0.01) as cancel_scope:
            await node.discover_peers()
        return cancel_scope.cancelled_caught

    assert trio.run(cancel_discovery) is True
    assert state.browsers[0].cancel_calls == 1
    assert state.zeroconfs[0].close_calls == 1


def test_repeated_discovery_balances_every_browser_and_zeroconf(monkeypatch):
    state = _install_fake_zeroconf(monkeypatch)
    monkeypatch.setenv("MCPPP_P2P_MDNS", "1")
    node = _node_with_identity()

    async def no_sleep(_delay):
        return None

    async def repeat():
        for _ in range(25):
            assert await node.discover_peers() == []

    monkeypatch.setattr(trio, "sleep", no_sleep)
    trio.run(repeat)

    assert len(state.browsers) == 25
    assert all(browser.cancel_calls == 1 for browser in state.browsers)
    assert len(state.zeroconfs) == 25
    assert all(zeroconf.close_calls == 1 for zeroconf in state.zeroconfs)


def test_discovery_rejects_self_and_does_not_store_candidates(monkeypatch):
    service_type = "_mcp-accelerate._tcp.local."
    state = _install_fake_zeroconf(
        monkeypatch,
        advertised_names=(
            f"{SELF_PEER_ID}.{service_type}",
            f"{REMOTE_PEER_ID}.{service_type}",
        ),
    )
    monkeypatch.setenv("MCPPP_P2P_MDNS", "1")

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(trio, "sleep", no_sleep)
    node = _node_with_identity()
    result = trio.run(node.discover_peers)

    assert [peer.peer_id for peer in result] == [REMOTE_PEER_ID]
    assert result[0].multiaddrs == [f"/ip4/10.8.0.99/tcp/19001/p2p/{REMOTE_PEER_ID}"]
    assert node._peers == {}
    assert state.browsers[0].cancel_calls == 1
    assert state.zeroconfs[0].close_calls == 1


def test_bootstrap_rejects_local_peer_without_dialing(monkeypatch):
    node = _node_with_identity()
    dialed = []

    class FakeHost:
        @staticmethod
        def get_id():
            return SELF_PEER_ID

        @staticmethod
        def get_addrs():
            return []

        async def connect(self, peer_info):
            dialed.append(peer_info)

    node._host = FakeHost()
    monkeypatch.setattr(
        "ipfs_accelerate_py.mcplusplus_module.p2p_transport.peerinfo_from_multiaddr",
        lambda _address: SimpleNamespace(peer_id=SELF_PEER_ID),
    )

    connected = trio.run(
        node._connect_bootstrap,
        f"/ip4/127.0.0.1/tcp/19001/p2p/{SELF_PEER_ID}",
    )

    assert connected is False
    assert dialed == []
    assert node._peers == {}
