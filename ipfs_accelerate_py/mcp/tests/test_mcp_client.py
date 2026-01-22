#!/usr/bin/env python3
"""
Test Script for IPFS Accelerate MCP Server

This script tests the MCP server by making direct calls to its tools and resources.
"""
import sys
import os
import json
import requests
import asyncio
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_test")

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test the IPFS Accelerate MCP server")
    parser.add_argument("--host", default="http://localhost:9999", help="Host URL")
    parser.add_argument("--mcp-path", default="/mcp", help="MCP server path")
    args = parser.parse_args()
    
    # Prepare URLs
    base_url = f"{args.host}{args.mcp_path}"
    logger.info(f"Testing MCP server at {base_url}")
    
    # Test server info
    try:
        logger.info("Getting server info...")
        response = requests.get(base_url)
        if response.status_code == 200:
            server_info = response.json()
            logger.info(f"Server name: {server_info.get('name')}")
            logger.info(f"Description: {server_info.get('description')}")
            logger.info(f"Version: {server_info.get('version')}")
        else:
            logger.error(f"Failed to get server info: {response.status_code}")
            logger.error(response.text)
            return
            
        # Test server health
        logger.info("Checking server health...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_info = response.json()
            logger.info(f"Health status: {health_info.get('status')}")
            logger.info(f"Tools count: {health_info.get('tools')}")
            logger.info(f"Resources count: {health_info.get('resources')}")
        else:
            logger.warning(f"Health check failed: {response.status_code}")
            
        # Test hardware detection tool
        logger.info("Testing hardware detection...")
        response = requests.post(f"{base_url}/tools/test_hardware", json={})
        if response.status_code == 200:
            hw_info = response.json()
            logger.info(f"Available accelerators: {hw_info.get('available_accelerators')}")
        else:
            logger.error(f"Hardware detection failed: {response.status_code}")
            logger.error(response.text)
            
        # Test hardware info tool
        logger.info("Getting hardware info...")
        response = requests.post(f"{base_url}/tools/get_hardware_info", json={})
        if response.status_code == 200:
            hw_info = response.json()
            logger.info(f"CPU: {hw_info.get('processor')}")
            logger.info(f"CUDA available: {hw_info.get('cuda_available')}")
        else:
            logger.error(f"Hardware info failed: {response.status_code}")
            logger.error(response.text)
            
        # Test inference if possible
        model = "BAAI/bge-small-en-v1.5"  # Example model
        logger.info(f"Testing inference with model {model}...")
        
        response = requests.post(
            f"{base_url}/tools/run_inference", 
            json={
                "model": model,
                "input_data": "This is a test input for embedding model"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Inference successful: {not result.get('error', False)}")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result preview: {str(result)[:200]}...")
        else:
            logger.error(f"Inference failed: {response.status_code}")
            logger.error(response.text)
            
        logger.info("MCP server testing completed")
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to MCP server at {base_url}")
    except Exception as e:
        logger.error(f"Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
