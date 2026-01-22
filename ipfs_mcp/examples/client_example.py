"""
Example MCP Client for IPFS Accelerate

This script demonstrates how to connect to and use the IPFS Accelerate MCP server
from a client application.
"""
import argparse
import logging
import sys
import json
from fastmcp.client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_client")

async def main():
    """Main entry point for the example client."""
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate MCP Client Example"
    )
    
    # Define command-line arguments
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL (default: http://127.0.0.1:8000/mcp)"
    )
    parser.add_argument(
        "--action",
        choices=["info", "hardware", "models", "inference"],
        default="info",
        help="Action to perform (default: info)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create MCP client
    logger.info(f"Connecting to MCP server at {args.server}...")
    client = MCPClient(args.server)
    
    try:
        # Get server information
        server_info = await client.get_server_info()
        print("Server Information:")
        print(f"  Name: {server_info.get('name', 'Unknown')}")
        print(f"  Description: {server_info.get('description', 'N/A')}")
        
        # Perform the requested action
        if args.action == "info":
            # Get system information
            system_info = await client.get_resource("system://info")
            print("\nSystem Information:")
            print(json.dumps(system_info, indent=2))
            
            # Get system capabilities
            capabilities = await client.get_resource("system://capabilities")
            print("\nSystem Capabilities:")
            print(json.dumps(capabilities, indent=2))
            
        elif args.action == "hardware":
            # Call hardware detection tool
            hardware = await client.invoke_tool("detect_hardware")
            print("\nHardware Detection Results:")
            print(json.dumps(hardware, indent=2))
            
            # Get optimal hardware recommendation
            optimal = await client.invoke_tool("get_optimal_hardware", {"model_type": "text-generation"})
            print("\nOptimal Hardware for Text Generation:")
            print(json.dumps(optimal, indent=2))
            
        elif args.action == "models":
            # Get available models
            models = await client.get_resource("models://available")
            print("\nAvailable Models:")
            print(json.dumps(models, indent=2))
            
            # Get details for a specific model
            if models and len(models) > 0:
                model_id = models[0]["id"]
                model_details = await client.get_resource(f"models://details/{model_id}")
                print(f"\nModel Details for {model_id}:")
                print(json.dumps(model_details, indent=2))
        
        elif args.action == "inference":
            # Perform sample inference
            input_data = "This is a sample input for inference"
            result = await client.invoke_tool("run_inference", {
                "model": "text-generation-model",
                "input_data": input_data
            })
            print("\nInference Result:")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
