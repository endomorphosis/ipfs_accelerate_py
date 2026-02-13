"""
Tests for TrioMCPClient

This module tests the Trio-native MCP client implementation.
"""

import pytest
import trio
from unittest.mock import Mock, AsyncMock, patch

from ipfs_accelerate_py.mcplusplus_module.trio import (
    TrioMCPClient,
    ClientConfig,
    is_trio_context,
)


class TestClientConfig:
    """Tests for ClientConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig(server_url="http://localhost:8000/mcp")
        assert config.server_url == "http://localhost:8000/mcp"
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.headers is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClientConfig(
            server_url="http://example.com/mcp",
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
            headers={"Authorization": "Bearer token"},
        )
        assert config.server_url == "http://example.com/mcp"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.headers == {"Authorization": "Bearer token"}
    
    def test_invalid_server_url(self):
        """Test that empty server_url raises error."""
        with pytest.raises(ValueError, match="server_url is required"):
            ClientConfig(server_url="")
    
    def test_invalid_timeout(self):
        """Test that negative timeout raises error."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ClientConfig(server_url="http://localhost:8000", timeout=-1.0)
    
    def test_invalid_max_retries(self):
        """Test that negative max_retries raises error."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ClientConfig(server_url="http://localhost:8000", max_retries=-1)


class TestTrioMCPClient:
    """Tests for TrioMCPClient."""
    
    @pytest.mark.trio
    async def test_client_initialization(self):
        """Test basic client initialization."""
        client = TrioMCPClient("http://localhost:8000/mcp")
        assert client.config.server_url == "http://localhost:8000/mcp"
        assert not client.is_connected
    
    @pytest.mark.trio
    async def test_client_with_config(self):
        """Test client initialization with custom config."""
        config = ClientConfig(
            server_url="http://example.com/mcp",
            timeout=60.0,
        )
        client = TrioMCPClient("http://localhost:8000", config=config)
        assert client.config.server_url == "http://example.com/mcp"
        assert client.config.timeout == 60.0
    
    @pytest.mark.trio
    async def test_client_with_kwargs(self):
        """Test client initialization with kwargs."""
        client = TrioMCPClient(
            "http://localhost:8000/mcp",
            timeout=45.0,
            max_retries=5,
        )
        assert client.config.timeout == 45.0
        assert client.config.max_retries == 5
    
    @pytest.mark.trio
    async def test_client_in_trio_context(self):
        """Test that client operations run in Trio context."""
        assert is_trio_context()
        
        client = TrioMCPClient("http://localhost:8000/mcp")
        assert is_trio_context()  # Should still be in Trio
    
    @pytest.mark.trio
    async def test_client_connect_close(self):
        """Test client connection lifecycle."""
        client = TrioMCPClient("http://localhost:8000/mcp")
        
        assert not client.is_connected
        
        # Mock httpx.AsyncClient to avoid actual network calls
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            await client.connect()
            assert client.is_connected
            
            await client.close()
            assert not client.is_connected
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.trio
    async def test_client_context_manager(self):
        """Test client as context manager."""
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            async with TrioMCPClient("http://localhost:8000/mcp") as client:
                assert client.is_connected
            
            # After context exit, should be closed
            assert not client.is_connected
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.trio
    async def test_call_tool_not_connected(self):
        """Test that calling tool when not connected raises error."""
        client = TrioMCPClient("http://localhost:8000/mcp")
        
        with pytest.raises(RuntimeError, match="Client not connected"):
            await client.call_tool("some_tool")
    
    @pytest.mark.trio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            # Setup mock response
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True, "result": "success"}
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            async with TrioMCPClient("http://localhost:8000/mcp") as client:
                result = await client.call_tool("test_tool", {"arg": "value"})
                
                assert result == {"ok": True, "result": "success"}
                mock_client.post.assert_called_once()
    
    @pytest.mark.trio
    async def test_call_tool_without_retry(self):
        """Test tool call without retry on failure."""
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            import httpx
            
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("Network error")
            mock_client_class.return_value = mock_client
            
            async with TrioMCPClient("http://localhost:8000/mcp") as client:
                with pytest.raises(httpx.HTTPError):
                    await client.call_tool("test_tool", retry=False)
                
                # Should only try once
                assert mock_client.post.call_count == 1
    
    @pytest.mark.trio
    async def test_list_tools(self):
        """Test listing available tools."""
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.json.return_value = [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"},
            ]
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            async with TrioMCPClient("http://localhost:8000/mcp") as client:
                tools = await client.list_tools()
                
                assert len(tools) == 2
                assert tools[0]["name"] == "tool1"
                mock_client.get.assert_called_once_with("/tools")
    
    @pytest.mark.trio
    async def test_get_server_info(self):
        """Test getting server information."""
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.json.return_value = {
                "name": "test-server",
                "version": "1.0.0",
            }
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            async with TrioMCPClient("http://localhost:8000/mcp") as client:
                info = await client.get_server_info()
                
                assert info["name"] == "test-server"
                assert info["version"] == "1.0.0"
                mock_client.get.assert_called_once_with("/")


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.mark.trio
    async def test_call_tool_convenience_function(self):
        """Test the call_tool convenience function."""
        from ipfs_accelerate_py.mcplusplus_module.trio.client import call_tool
        
        with patch('ipfs_accelerate_py.mcplusplus_module.trio.client.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True}
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await call_tool(
                "http://localhost:8000/mcp",
                "test_tool",
                {"arg": "value"}
            )
            
            assert result == {"ok": True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
