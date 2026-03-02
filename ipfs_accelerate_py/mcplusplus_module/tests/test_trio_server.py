"""
Tests for TrioMCPServer

This module tests the Trio-native MCP server implementation.
"""

import pytest
import trio
from types import SimpleNamespace
from unittest.mock import Mock

from ipfs_accelerate_py.mcplusplus_module.trio import (
    TrioMCPServer,
    ServerConfig,
    is_trio_context,
)


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

    def test_resolve_p2p_registrars_returns_callables(self):
        """Resolver should return callable taskqueue/workflow registrars."""
        server = TrioMCPServer()
        taskqueue_registrar, workflow_registrar = server._resolve_p2p_registrars()
        assert callable(taskqueue_registrar)
        assert callable(workflow_registrar)

    def test_resolve_p2p_registrars_prefers_canonical_modules(self, monkeypatch):
        """Resolver should prefer explicit module imports when available."""
        from ipfs_accelerate_py.mcplusplus_module.trio import server as server_module

        calls = []

        def _taskqueue(_mcp):
            return None

        def _workflow(_mcp):
            return None

        def _fake_import_module(name: str):
            calls.append(name)
            if name.endswith("tools.taskqueue_tools"):
                return SimpleNamespace(register_p2p_taskqueue_tools=_taskqueue)
            if name.endswith("tools.workflow_tools"):
                return SimpleNamespace(register_p2p_workflow_tools=_workflow)
            raise AssertionError(f"Unexpected import: {name}")

        monkeypatch.setattr(server_module, "import_module", _fake_import_module)

        server = TrioMCPServer()
        taskqueue_registrar, workflow_registrar = server._resolve_p2p_registrars()

        assert taskqueue_registrar is _taskqueue
        assert workflow_registrar is _workflow
        assert calls == [
            "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools",
            "ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools",
        ]

    def test_resolve_p2p_registrars_falls_back_to_package(self, monkeypatch):
        """Resolver should fall back to package-level registrars on import failure."""
        from ipfs_accelerate_py.mcplusplus_module.trio import server as server_module

        calls = []

        def _taskqueue(_mcp):
            return None

        def _workflow(_mcp):
            return None

        def _fake_import_module(name: str):
            calls.append(name)
            if name.endswith("tools.taskqueue_tools"):
                raise ImportError("simulated module import failure")
            if name.endswith("mcplusplus_module.tools"):
                return SimpleNamespace(
                    register_p2p_taskqueue_tools=_taskqueue,
                    register_p2p_workflow_tools=_workflow,
                )
            raise AssertionError(f"Unexpected import: {name}")

        monkeypatch.setattr(server_module, "import_module", _fake_import_module)

        server = TrioMCPServer()
        taskqueue_registrar, workflow_registrar = server._resolve_p2p_registrars()

        assert taskqueue_registrar is _taskqueue
        assert workflow_registrar is _workflow
        assert calls == [
            "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools",
            "ipfs_accelerate_py.mcplusplus_module.tools",
        ]

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

    def test_register_p2p_tools_uses_explicit_registrars(self, monkeypatch):
        """Both registrars should be called when both feature flags are enabled."""
        from ipfs_accelerate_py.mcplusplus_module.tools import taskqueue_tools as taskqueue_module
        from ipfs_accelerate_py.mcplusplus_module.tools import workflow_tools as workflow_module

        calls = []

        def _register_taskqueue(_mcp):
            calls.append("taskqueue")

        def _register_workflow(_mcp):
            calls.append("workflow")

        monkeypatch.setattr(taskqueue_module, "register_p2p_taskqueue_tools", _register_taskqueue)
        monkeypatch.setattr(workflow_module, "register_p2p_workflow_tools", _register_workflow)

        server = TrioMCPServer(ServerConfig(enable_p2p_tools=True, enable_taskqueue_tools=True, enable_workflow_tools=True))
        server.mcp = Mock()

        server._register_p2p_tools()
        assert calls == ["taskqueue", "workflow"]

    def test_register_p2p_tools_respects_feature_flags(self, monkeypatch):
        """Only enabled registrar should be called when one feature flag is disabled."""
        from ipfs_accelerate_py.mcplusplus_module.tools import taskqueue_tools as taskqueue_module
        from ipfs_accelerate_py.mcplusplus_module.tools import workflow_tools as workflow_module

        calls = []

        def _register_taskqueue(_mcp):
            calls.append("taskqueue")

        def _register_workflow(_mcp):
            calls.append("workflow")

        monkeypatch.setattr(taskqueue_module, "register_p2p_taskqueue_tools", _register_taskqueue)
        monkeypatch.setattr(workflow_module, "register_p2p_workflow_tools", _register_workflow)

        server = TrioMCPServer(ServerConfig(enable_p2p_tools=True, enable_taskqueue_tools=False, enable_workflow_tools=True))
        server.mcp = Mock()

        server._register_p2p_tools()
        assert calls == ["workflow"]
    
    @pytest.mark.trio
    async def test_server_setup(self):
        """Test server setup process."""
        config = ServerConfig(enable_p2p_tools=False)  # Disable to avoid dependency issues
        server = TrioMCPServer(config=config)
        
        # Setup should create mcp instance
        server.setup()
        assert server.mcp is not None
        assert server.fastapi_app is not None
    
    @pytest.mark.trio
    async def test_server_in_trio_context(self):
        """Test that server operations run in Trio context."""
        assert is_trio_context()
        
        server = TrioMCPServer()
        assert is_trio_context()  # Should still be in Trio
    
    @pytest.mark.trio
    async def test_server_lifecycle_hooks(self):
        """Test server startup and shutdown hooks."""
        config = ServerConfig(enable_p2p_tools=False)
        server = TrioMCPServer(config=config)
        server.setup()
        
        # Test startup
        await server._startup()
        assert server._started is True
        
        # Test shutdown
        await server._shutdown()
        assert server._started is False
    
    @pytest.mark.trio
    async def test_server_run_with_timeout(self):
        """Test server run with timeout (to avoid infinite run)."""
        config = ServerConfig(enable_p2p_tools=False)
        server = TrioMCPServer(config=config)
        
        # Run with a timeout to ensure it starts but doesn't run forever
        with trio.move_on_after(0.1) as cancel_scope:
            await server.run()
        
        # Should have been cancelled by timeout
        assert cancel_scope.cancelled_caught
        assert server._started is False  # Should be shut down


class TestServerIntegration:
    """Integration tests for server functionality."""
    
    @pytest.mark.trio
    async def test_create_asgi_app(self):
        """Test ASGI app creation."""
        config = ServerConfig(enable_p2p_tools=False)
        server = TrioMCPServer(config=config)
        
        app = server.create_asgi_app()
        assert app is not None
        assert hasattr(app, 'routes') or hasattr(app, 'router')
    
    @pytest.mark.trio
    async def test_server_with_nursery(self):
        """Test running server within a nursery."""
        config = ServerConfig(enable_p2p_tools=False)
        server = TrioMCPServer(config=config)
        
        async with trio.open_nursery() as nursery:
            # Start server in background
            nursery.start_soon(server.run)
            
            # Give it a moment to start
            await trio.sleep(0.05)
            
            # Cancel the nursery to stop the server
            nursery.cancel_scope.cancel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
