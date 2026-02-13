"""
Tests for TrioMCPServer

This module tests the Trio-native MCP server implementation.
"""

import pytest
import trio

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
