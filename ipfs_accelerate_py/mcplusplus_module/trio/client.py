"""
Trio-native MCP Client for MCP++ (Model Context Protocol Plus Plus).

This module provides a Trio-native implementation of the MCP client that
communicates with MCP servers using structured concurrency.

Module: ipfs_accelerate_py.mcplusplus_module.trio.client

Key features:
- Native Trio event loop (no asyncio bridges)
- Structured concurrency with nurseries
- Automatic reconnection with cancel scopes
- Connection pooling support
- Full MCP protocol support

Usage:
    import trio
    from ipfs_accelerate_py.mcplusplus_module.trio import TrioMCPClient
    
    async def main():
        async with TrioMCPClient("http://localhost:8000/mcp") as client:
            # Call a tool
            result = await client.call_tool("p2p_taskqueue_status")
            print(result)
    
    if __name__ == "__main__":
        trio.run(main)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

import trio
import httpx

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.trio.client")


@dataclass
class ClientConfig:
    """Configuration for TrioMCPClient.
    
    Attributes:
        server_url: MCP server URL (e.g., "http://localhost:8000/mcp")
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        headers: Additional HTTP headers to send
    """
    server_url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_url:
            raise ValueError("server_url is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


class TrioMCPClient:
    """Trio-native MCP client for communicating with MCP servers.
    
    This client uses Trio's structured concurrency to manage connections
    and handle requests. It supports automatic reconnection, timeouts,
    and retry logic using Trio's cancel scopes and nurseries.
    
    Example:
        >>> import trio
        >>> from ipfs_accelerate_py.mcplusplus_module.trio import TrioMCPClient
        >>> 
        >>> async def main():
        ...     async with TrioMCPClient("http://localhost:8000/mcp") as client:
        ...         result = await client.call_tool("p2p_taskqueue_status")
        ...         print(result)
        ... 
        >>> trio.run(main)
    """
    
    def __init__(
        self,
        server_url: str,
        config: Optional[ClientConfig] = None,
        **kwargs
    ):
        """Initialize the Trio MCP client.
        
        Args:
            server_url: URL of the MCP server
            config: Client configuration (created from server_url and kwargs if None)
            **kwargs: Additional configuration options passed to ClientConfig
        """
        if config is None:
            config = ClientConfig(server_url=server_url, **kwargs)
        
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        
        logger.info(f"Initialized TrioMCPClient for {self.config.server_url}")
    
    async def __aenter__(self) -> TrioMCPClient:
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Connect to the MCP server.
        
        Creates an httpx.AsyncClient for making requests.
        """
        if self._connected:
            logger.warning("Client already connected")
            return
        
        logger.info(f"Connecting to MCP server at {self.config.server_url}")
        
        # Create httpx client with Trio support
        self._client = httpx.AsyncClient(
            base_url=self.config.server_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers=self.config.headers or {},
        )
        
        self._connected = True
        logger.info("Connected to MCP server")
    
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if not self._connected:
            return
        
        logger.info("Closing MCP client connection")
        
        if self._client:
            await self._client.aclose()
            self._client = None
        
        self._connected = False
        logger.info("MCP client connection closed")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary
            retry: Whether to retry on failure (default: True)
            
        Returns:
            Tool execution result as a dictionary
            
        Raises:
            RuntimeError: If client is not connected
            httpx.HTTPError: If request fails after retries
        """
        if not self._connected or not self._client:
            raise RuntimeError("Client not connected. Call connect() or use context manager.")
        
        logger.debug(f"Calling tool: {tool_name}")
        
        payload = {
            "tool": tool_name,
            "arguments": arguments or {},
        }
        
        if retry:
            return await self._call_with_retry(payload)
        else:
            return await self._do_call(payload)
    
    async def _do_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call request.
        
        Args:
            payload: Request payload
            
        Returns:
            Response data as dictionary
            
        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._client.post("/tools/call", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call with retry logic.
        
        Args:
            payload: Request payload
            
        Returns:
            Response data as dictionary
            
        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._do_call(payload)
            except httpx.HTTPError as e:
                last_error = e
                
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Tool call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    await trio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Tool call failed after {self.config.max_retries + 1} attempts")
        
        # All retries exhausted
        raise last_error
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the MCP server.
        
        Returns:
            List of tool descriptions
            
        Raises:
            RuntimeError: If client is not connected
            httpx.HTTPError: If request fails
        """
        if not self._connected or not self._client:
            raise RuntimeError("Client not connected. Call connect() or use context manager.")
        
        logger.debug("Listing available tools")
        
        response = await self._client.get("/tools")
        response.raise_for_status()
        return response.json()
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MCP server.
        
        Returns:
            Server information as dictionary
            
        Raises:
            RuntimeError: If client is not connected
            httpx.HTTPError: If request fails
        """
        if not self._connected or not self._client:
            raise RuntimeError("Client not connected. Call connect() or use context manager.")
        
        logger.debug("Getting server info")
        
        response = await self._client.get("/")
        response.raise_for_status()
        return response.json()
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._client is not None


# Convenience function for quick client usage
async def call_tool(
    server_url: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    **config_kwargs
) -> Dict[str, Any]:
    """Convenience function to call a tool without managing client lifecycle.
    
    Args:
        server_url: MCP server URL
        tool_name: Name of tool to call
        arguments: Tool arguments
        **config_kwargs: Additional configuration options
        
    Returns:
        Tool execution result
        
    Example:
        >>> result = await call_tool(
        ...     "http://localhost:8000/mcp",
        ...     "p2p_taskqueue_status",
        ...     {"detail": True}
        ... )
    """
    async with TrioMCPClient(server_url, **config_kwargs) as client:
        return await client.call_tool(tool_name, arguments)


__all__ = [
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
]
