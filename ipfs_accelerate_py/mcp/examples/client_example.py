#!/usr/bin/env python3
"""
IPFS Accelerate MCP Client Example

This script demonstrates how to use the IPFS Accelerate MCP client to interact with
an MCP server. It shows how to use tools and access resources.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional, List, Union
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.examples.client")

def pretty_print_json(data: Dict[str, Any]) -> None:
    """
    Pretty print JSON data
    
    Args:
        data: Dictionary to print
    """
    print(json.dumps(data, indent=2, sort_keys=True))

def example_hardware_tools(base_url: str) -> None:
    """
    Demonstrate the hardware tools
    
    Args:
        base_url: Base URL of the MCP server
    """
    print("\n=== Hardware Tools Example ===\n")
    
    try:
        import requests
        
        # Get hardware information
        print("Getting hardware information...\n")
        
        response = requests.post(f"{base_url}/tools/get_hardware_info", json={
            "include_detailed": True
        })
        
        if response.status_code == 200:
            hardware_info = response.json()
            print("Hardware information:")
            pretty_print_json(hardware_info)
        else:
            print(f"Error getting hardware information: {response.status_code}")
            print(response.text)
        
        # Test hardware
        print("\nTesting hardware...\n")
        
        response = requests.post(f"{base_url}/tools/test_hardware", json={
            "tests": ["cpu", "cuda", "webgpu"],
            "include_benchmarks": True
        })
        
        if response.status_code == 200:
            test_results = response.json()
            print("Hardware test results:")
            pretty_print_json(test_results)
        else:
            print(f"Error testing hardware: {response.status_code}")
            print(response.text)
        
        # Get hardware recommendations
        print("\nGetting hardware recommendations...\n")
        
        response = requests.post(f"{base_url}/tools/recommend_hardware", json={
            "model_name": "llama-7b",
            "model_type": "llm",
            "include_available_only": True
        })
        
        if response.status_code == 200:
            recommendations = response.json()
            print("Hardware recommendations:")
            pretty_print_json(recommendations)
        else:
            print(f"Error getting hardware recommendations: {response.status_code}")
            print(response.text)
    
    except ImportError:
        print("The requests package is required to run this example.")
        print("Please install it with 'pip install requests'.")
    
    except Exception as e:
        print(f"Error in hardware tools example: {e}")

def example_resources(base_url: str) -> None:
    """
    Demonstrate the resources
    
    Args:
        base_url: Base URL of the MCP server
    """
    print("\n=== Resources Example ===\n")
    
    try:
        import requests
        
        # Get version information
        print("Getting version information...\n")
        
        response = requests.get(f"{base_url}/resources/ipfs_accelerate/version")
        
        if response.status_code == 200:
            version_info = response.json()
            print("Version information:")
            pretty_print_json(version_info)
        else:
            print(f"Error getting version information: {response.status_code}")
            print(response.text)
        
        # Get system information
        print("\nGetting system information...\n")
        
        response = requests.get(f"{base_url}/resources/ipfs_accelerate/system_info")
        
        if response.status_code == 200:
            system_info = response.json()
            print("System information:")
            pretty_print_json(system_info)
        else:
            print(f"Error getting system information: {response.status_code}")
            print(response.text)
        
        # Get supported models
        print("\nGetting supported models...\n")
        
        response = requests.get(f"{base_url}/resources/ipfs_accelerate/supported_models")
        
        if response.status_code == 200:
            supported_models = response.json()
            print("Supported models:")
            pretty_print_json(supported_models)
        else:
            print(f"Error getting supported models: {response.status_code}")
            print(response.text)
    
    except ImportError:
        print("The requests package is required to run this example.")
        print("Please install it with 'pip install requests'.")
    
    except Exception as e:
        print(f"Error in resources example: {e}")

def main() -> None:
    """
    Main entry point for the client example
    
    This function parses command-line arguments and runs the client example.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Client Example")
    
    parser.add_argument("--host", default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8000, help="MCP server port")
    parser.add_argument("--mount-path", default="/mcp", help="MCP server mount path")
    parser.add_argument("--examples", nargs="+", default=["hardware", "resources"], 
                        choices=["hardware", "resources", "all"],
                        help="Examples to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up base URL
    base_url = f"http://{args.host}:{args.port}{args.mount_path}"
    
    print(f"IPFS Accelerate MCP Client Example")
    print(f"MCP Server: {base_url}")
    
    # Run examples
    examples = args.examples
    if "all" in examples:
        examples = ["hardware", "resources"]
    
    try:
        # Run hardware tools example
        if "hardware" in examples:
            example_hardware_tools(base_url)
        
        # Run resources example
        if "resources" in examples:
            example_resources(base_url)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping example...")
    
    except Exception as e:
        print(f"Error in client example: {e}")

if __name__ == "__main__":
    main()
