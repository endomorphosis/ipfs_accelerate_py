"""
Tests for TrioMCPServer

This module tests the Trio-native MCP server implementation.
"""

import pytest
import anyio
import trio
from unittest.mock import Mock

from ipfs_accelerate_py.mcplusplus_module.trio import (
    TrioMCPServer,
    ServerConfig,
    is_trio_context,
)
from ipfs_accelerate_py.mcplusplus_module.p2p_transport import (
    MCPp2pNode,
    PeerInfo,
)

pytestmark = pytest.mark.anyio


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.name == "ipfs-accelerate-mcp-trio"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.mount_path == "/mcp"
        assert config.debug is False
        assert config.enable_p2p_tools is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ServerConfig(
            name="test-server",
            host="127.0.0.1",
            port=9000,
            mount_path="/api",
            debug=True,
            enable_p2p_tools=False,
        )
        assert config.name == "test-server"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.mount_path == "/api"
        assert config.debug is True
        assert config.enable_p2p_tools is False

    def test_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("MCP_SERVER_NAME", "env-server")
        monkeypatch.setenv("MCP_HOST", "localhost")
        monkeypatch.setenv("MCP_PORT", "8080")
        monkeypatch.setenv("MCP_DEBUG", "1")

        config = ServerConfig.from_env()
        assert config.name == "env-server"
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.debug is True


class TestTrioMCPServer:
    """Tests for TrioMCPServer."""

    def test_server_initialization(self):
        """Test basic server initialization."""
        server = TrioMCPServer()
        assert server.config.name == "ipfs-accelerate-mcp-trio"
        assert server.mcp is None
        assert server.fastapi_app is None
        assert server._started is False

    def test_server_with_custom_config(self):
        """Test server initialization with custom config."""
        config = ServerConfig(name="test-server", port=9000)
        server = TrioMCPServer(config=config)
        assert server.config.name == "test-server"
        assert server.config.port == 9000

    def test_server_with_name_override(self):
        """Test server initialization with name override."""
        config = ServerConfig(name="config-name")
        server = TrioMCPServer(config=config, name="override-name")
        assert server.config.name == "override-name"

    def test_maintenance_skips_disabled_mdns_and_retries_bootstrap(self, monkeypatch):
        """Disabled mDNS must not suppress configured bootstrap retries."""
        bootstrap = "/ip4/10.8.0.99/tcp/19001/p2p/12D3KooWBootstrap"
        server = TrioMCPServer()
        server._started = True
        node = MCPp2pNode(bootstrap_peers=[bootstrap])
        calls = []
        discovery_calls = []

        async def no_sleep(_delay):
            return None

        async def unexpected_discovery():
            discovery_calls.append(True)
            raise AssertionError("mDNS discovery must be skipped when disabled")

        async def connect(peer_addr):
            calls.append(peer_addr)
            server._started = False
            return True

        monkeypatch.setenv("MCPPP_P2P_MDNS", "0")
        monkeypatch.setattr(trio, "sleep", no_sleep)
        monkeypatch.setattr(node, "discover_peers", unexpected_discovery)
        monkeypatch.setattr(node, "_connect_bootstrap", connect)

        trio.run(server._p2p_maintenance_loop, node)

        assert discovery_calls == []
        assert calls == [bootstrap]

    def test_self_only_discovery_does_not_suppress_bootstrap(self, monkeypatch):
        """A legacy/local discovery record must not count as a remote peer."""
        bootstrap = "/ip4/10.8.0.99/tcp/19001/p2p/12D3KooWBootstrap"
        server = TrioMCPServer()
        server._started = True
        node = MCPp2pNode(bootstrap_peers=[bootstrap])
        node._host = Mock()
        node._host.get_id.return_value = "12D3KooWLocal"
        calls = []

        async def no_sleep(_delay):
            return None

        async def discover_self():
            peer = PeerInfo(
                peer_id=node.peer_id,
                multiaddrs=[f"/ip4/10.8.0.99/tcp/19001/p2p/{node.peer_id}"],
            )
            # Simulate state left by the previous discover_peers implementation.
            node._peers[node.peer_id] = peer
            server._started = False
            return [peer]

        async def connect(peer_addr):
            calls.append(peer_addr)
            return True

        monkeypatch.setenv("MCPPP_P2P_MDNS", "1")
        monkeypatch.setattr(trio, "sleep", no_sleep)
        monkeypatch.setattr(node, "discover_peers", discover_self)
        monkeypatch.setattr(node, "_connect_bootstrap", connect)

        trio.run(server._p2p_maintenance_loop, node)

        assert calls == [bootstrap]
        assert node.peer_id not in node._peers

    def test_resolve_p2p_registrars_returns_callables(self):
        """Resolver should return callable taskqueue/workflow registrars."""
        server = TrioMCPServer()
        taskqueue_registrar, workflow_registrar = server._resolve_p2p_registrars()
        assert callable(taskqueue_registrar)
        assert callable(workflow_registrar)

    def test_resolve_p2p_registrars_delegates_to_tools_resolver(self, monkeypatch):
        """Trio resolver should delegate to the shared tools resolver path."""
        from ipfs_accelerate_py.mcplusplus_module import tools as tools_module
        from ipfs_accelerate_py.mcp_server import compatibility as canonical_compat

        assert tools_module._resolve_p2p_registrars is canonical_compat._resolve_p2p_registrars

        def _taskqueue(_mcp):
            return None

        def _workflow(_mcp):
            return None

        monkeypatch.setattr(
            tools_module,
            "_resolve_p2p_registrars",
            lambda: (_taskqueue, _workflow),
        )

        server = TrioMCPServer()
        taskqueue_registrar, workflow_registrar = server._resolve_p2p_registrars()

        assert taskqueue_registrar is _taskqueue
        assert workflow_registrar is _workflow

    def test_register_p2p_tools_uses_resolved_registrars(self, monkeypatch):
        """Registration should execute callables returned by resolver hook."""
        calls = []

        def _register_taskqueue(_mcp):
            calls.append("taskqueue")

        def _register_workflow(_mcp):
            calls.append("workflow")

        server = TrioMCPServer(
            ServerConfig(
                enable_p2p_tools=True,
                enable_taskqueue_tools=True,
                enable_workflow_tools=True,
            )
        )
        server.mcp = Mock()

        monkeypatch.setattr(
            server,
            "_resolve_p2p_registrars",
            lambda: (_register_taskqueue, _register_workflow),
        )

        server._register_p2p_tools()
        assert calls == ["taskqueue", "workflow"]

    def test_register_p2p_tools_adapts_native_registrars_to_standalone(self):
        """Hierarchical native registrars should populate the standalone MCP registry."""
        from ipfs_accelerate_py.mcp_server.server import StandaloneMCP

        server = TrioMCPServer()
        server.mcp = StandaloneMCP(name="test-mcplusplus")

        server._register_p2p_tools()

        assert "p2p_taskqueue_status" in server.mcp.tools
        assert "get_p2p_scheduler_status" in server.mcp.tools
        assert callable(server.mcp.tools["p2p_taskqueue_status"]["function"])

    def test_register_p2p_tools_uses_explicit_registrars(self, monkeypatch):
        """Canonical explicit registrars should be called when both feature flags are enabled."""
        from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools as taskqueue_module
        from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools import (
            native_p2p_workflow_tools as workflow_module,
        )

        calls = []

        def _register_taskqueue(_mcp):
            calls.append("taskqueue")

        def _register_workflow(_mcp):
            calls.append("workflow")

        monkeypatch.setattr(taskqueue_module, "register_native_p2p_tools", _register_taskqueue)
        monkeypatch.setattr(
            workflow_module, "register_native_p2p_workflow_tools", _register_workflow
        )

        server = TrioMCPServer(
            ServerConfig(
                enable_p2p_tools=True, enable_taskqueue_tools=True, enable_workflow_tools=True
            )
        )
        server.mcp = Mock()

        server._register_p2p_tools()
        assert calls == ["taskqueue", "workflow"]

    def test_register_p2p_tools_respects_feature_flags(self, monkeypatch):
        """Only enabled canonical registrar should be called when one feature flag is disabled."""
        from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools as taskqueue_module
        from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools import (
            native_p2p_workflow_tools as workflow_module,
        )

        calls = []

        def _register_taskqueue(_mcp):
            calls.append("taskqueue")

        def _register_workflow(_mcp):
            calls.append("workflow")

        monkeypatch.setattr(taskqueue_module, "register_native_p2p_tools", _register_taskqueue)
        monkeypatch.setattr(
            workflow_module, "register_native_p2p_workflow_tools", _register_workflow
        )

        server = TrioMCPServer(
            ServerConfig(
                enable_p2p_tools=True, enable_taskqueue_tools=False, enable_workflow_tools=True
            )
        )
        server.mcp = Mock()

        server._register_p2p_tools()
        assert calls == ["workflow"]

    def test_server_setup(self):
        """Test server setup process."""

        async def _run() -> None:
            config = ServerConfig(enable_p2p_tools=False)  # Disable to avoid dependency issues
            server = TrioMCPServer(config=config)
            server.setup()
            assert server.mcp is not None
            assert server.fastapi_app is not None

        anyio.run(_run)

    def test_server_in_trio_context(self):
        """Test that server operations run in Trio context."""

        async def _run() -> None:
            assert is_trio_context()
            _server = TrioMCPServer()
            assert is_trio_context()  # Should still be in Trio

        anyio.run(_run)

    def test_server_lifecycle_hooks(self):
        """Test server startup and shutdown hooks."""

        async def _run() -> None:
            config = ServerConfig(enable_p2p_tools=False)
            server = TrioMCPServer(config=config)
            server.setup()
            await server._startup()
            assert server._started is True
            await server._shutdown()
            assert server._started is False

        anyio.run(_run)

    def test_server_run_with_timeout(self):
        """Test server run with timeout (to avoid infinite run)."""

        async def _run() -> None:
            config = ServerConfig(enable_p2p_tools=False)
            server = TrioMCPServer(config=config)
            with trio.move_on_after(0.1) as cancel_scope:
                await server.run()
            assert cancel_scope.cancelled_caught
            assert server._started is False  # Should be shut down

        anyio.run(_run)


class TestServerIntegration:
    """Integration tests for server functionality."""

    def test_create_asgi_app(self):
        """Test ASGI app creation."""

        async def _run() -> None:
            config = ServerConfig(enable_p2p_tools=False)
            server = TrioMCPServer(config=config)
            app = server.create_asgi_app()
            assert app is not None
            assert hasattr(app, "routes") or hasattr(app, "router")

        anyio.run(_run)

    def test_server_with_nursery(self):
        """Test running server within a nursery."""

        async def _run() -> None:
            config = ServerConfig(enable_p2p_tools=False)
            server = TrioMCPServer(config=config)
            async with trio.open_nursery() as nursery:
                nursery.start_soon(server.run)
                await trio.sleep(0.05)
                nursery.cancel_scope.cancel()

        anyio.run(_run)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
