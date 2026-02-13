"""
Example: Running the Trio-native MCP Server

This example demonstrates how to run the TrioMCPServer in different modes.
"""

import os
import trio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def example_with_timeout():
    """Example: Run server with a timeout.
    
    Useful for testing or running the server for a limited time.
    """
    from ipfs_accelerate_py.mcplusplus_module import TrioMCPServer, ServerConfig
    
    logger.info("=" * 60)
    logger.info("Example: Server with Timeout")
    logger.info("=" * 60)
    
    config = ServerConfig(
        name="timeout-server",
        host="127.0.0.1",
        port=8003,
        enable_p2p_tools=False,
    )
    
    server = TrioMCPServer(config=config)
    
    logger.info("Starting server with 2-second timeout")
    
    with trio.move_on_after(2) as cancel_scope:
        await server.run()
    
    if cancel_scope.cancelled_caught:
        logger.info("Server timed out as expected")
    else:
        logger.info("Server exited before timeout")


async def main():
    """Run example."""
    logger.info("\nTrio MCP Server Example\n")
    
    try:
        await example_with_timeout()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    
    logger.info("\nExample completed!")


if __name__ == "__main__":
    trio.run(main)
