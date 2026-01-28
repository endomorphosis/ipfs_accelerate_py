"""
IPFS network operations tools for the MCP server.

This module provides tools that expose IPFS network operations to LLM clients,
including peer discovery, pubsub, and DHT operations.
"""

import os
import json
import anyio
from typing import Dict, Any, List, Optional, cast

# Try imports with fallbacks
try:
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        raise ImportError("Using mock MCP under pytest")
    from fastmcp import FastMCP, Context
except ImportError:
    try:
        from mcp.mock_mcp import FastMCP, Context
    except ImportError:
        from mock_mcp import FastMCP, Context

# Import from the types module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.types import IPFSAccelerateContext

# Get IPFS client function (reusing from ipfs_files.py)
try:
    from .ipfs_files import get_ipfs_client
except ImportError:
    from tools.ipfs_files import get_ipfs_client


def register_network_tools(mcp: FastMCP) -> None:
    """Register IPFS network operation tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance to register tools with
    """
    
    @mcp.tool()
    async def ipfs_id(ctx: Context) -> Dict[str, Any]:
        """Get information about the local IPFS node.
        
        Args:
            ctx: MCP context
            
        Returns:
            Node identity information
        """
        await ctx.info("Getting IPFS node identity information")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Get node ID information
            result = await anyio.to_thread.run_sync(ipfs.id)
            
            return {
                "id": result.get("ID", ""),
                "addresses": result.get("Addresses", []),
                "agent_version": result.get("AgentVersion", ""),
                "protocol_version": result.get("ProtocolVersion", ""),
                "public_key": result.get("PublicKey", "")
            }
        except Exception as e:
            await ctx.error(f"Error getting node identity: {str(e)}")
            return {"error": str(e)}
    
    @mcp.tool()
    async def ipfs_swarm_peers(ctx: Context) -> List[Dict[str, Any]]:
        """List peers connected to the IPFS swarm.
        
        Args:
            ctx: MCP context
            
        Returns:
            List of peers with connection information
        """
        await ctx.info("Listing connected peers")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Get swarm peers
            result = await anyio.to_thread.run_sync(ipfs.swarm_peers)
            
            # Format the peers into a cleaner structure
            peers = []
            for peer in result.get("Peers", []):
                peer_info = {
                    "peer_id": peer.get("Peer", ""),
                    "addr": peer.get("Addr", ""),
                    "latency": peer.get("Latency", "")
                }
                peers.append(peer_info)
            
            return peers
        except Exception as e:
            await ctx.error(f"Error listing peers: {str(e)}")
            return [{"error": str(e)}]
    
    @mcp.tool()
    async def ipfs_swarm_connect(addr: str, ctx: Context) -> Dict[str, Any]:
        """Connect to a peer in the IPFS swarm.
        
        Args:
            addr: Multiaddress of the peer to connect to
            ctx: MCP context
            
        Returns:
            Connection status information
        """
        await ctx.info(f"Connecting to peer: {addr}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Connect to the peer
            result = await anyio.to_thread.run_sync(ipfs.swarm_connect, addr)
            
            return {
                "success": True,
                "address": addr,
                "message": result.get("Strings", ["Connection successful"])[0]
            }
        except Exception as e:
            await ctx.error(f"Error connecting to peer: {str(e)}")
            return {
                "success": False,
                "address": addr,
                "error": str(e)
            }
    
    @mcp.tool()
    async def ipfs_pubsub_pub(topic: str, message: str, ctx: Context) -> Dict[str, Any]:
        """Publish a message to an IPFS pubsub topic.
        
        Args:
            topic: The pubsub topic to publish to
            message: The message to publish
            ctx: MCP context
            
        Returns:
            Status of the publish operation
        """
        await ctx.info(f"Publishing message to topic: {topic}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Convert message to bytes
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message
            
            # Publish the message
            await anyio.to_thread.run_sync(ipfs.pubsub_pub, topic, message_bytes)
            
            return {
                "success": True,
                "topic": topic,
                "message_length": len(message_bytes)
            }
        except Exception as e:
            await ctx.error(f"Error publishing message: {str(e)}")
            return {
                "success": False,
                "topic": topic,
                "error": str(e)
            }
    
    @mcp.tool()
    async def ipfs_dht_findpeer(peer_id: str, ctx: Context) -> List[Dict[str, Any]]:
        """Find addresses for a peer in the IPFS DHT.
        
        Args:
            peer_id: The ID of the peer to find
            ctx: MCP context
            
        Returns:
            List of addresses for the peer
        """
        await ctx.info(f"Finding addresses for peer: {peer_id}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Find the peer
            result = await anyio.to_thread.run_sync(ipfs.dht_findpeer, peer_id)
            
            # Format the addresses
            addresses = []
            for addr in result.get("Responses", []):
                for address in addr.get("Addrs", []):
                    addresses.append({
                        "peer_id": peer_id,
                        "address": address
                    })
            
            return addresses
        except Exception as e:
            await ctx.error(f"Error finding peer: {str(e)}")
            return [{"peer_id": peer_id, "error": str(e)}]
    
    @mcp.tool()
    async def ipfs_dht_findprovs(cid: str, ctx: Context, num_providers: int = 20) -> List[Dict[str, Any]]:
        """Find providers for a CID in the IPFS DHT.
        
        Args:
            cid: The content identifier to find providers for
            ctx: MCP context
            num_providers: Maximum number of providers to find
            
        Returns:
            List of provider information
        """
        await ctx.info(f"Finding providers for CID: {cid}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client(ctx)
            
            # Find providers
            result = await anyio.to_thread.run_sync(ipfs.dht_findprovs, cid, num_providers=num_providers)
            
            # Format the providers
            providers = []
            for response in result.get("Responses", []):
                provider = {
                    "id": response.get("ID", ""),
                    "addresses": response.get("Addrs", [])
                }
                providers.append(provider)
            
            return providers
        except Exception as e:
            await ctx.error(f"Error finding providers: {str(e)}")
            return [{"error": str(e)}]


if __name__ == "__main__":
    # This can be used for standalone testing
    import os
    import sys
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from server import create_ipfs_mcp_server
    
    async def test_tools():
        mcp = create_ipfs_mcp_server("IPFS Network Tools Test")
        register_network_tools(mcp)
        # Implement test code here if needed
    
    anyio.run(test_tools)
