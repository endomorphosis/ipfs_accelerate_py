#!/usr/bin/env python3
"""
MCP Connection Verification Script

This script verifies that both MCP servers (IPFS and Direct) are properly running
and accessible through their respective endpoints. It can also install the updated
settings file for proper client configuration.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests
from typing import Dict, Any, List, Optional
import shutil

# Configuration
MCP_SERVERS = {
    "ipfs-accelerate-mcp": {
        "url": "http://localhost:8002",
        "sse_url": "http://localhost:8002/sse",
        "manifest_url": "http://localhost:8002/mcp/manifest",
        "description": "IPFS Core MCP Server"
    },
    "ipfs-direct-mcp": {
        "url": "http://localhost:8001",
        "sse_url": "http://localhost:8001/sse",
        "manifest_url": None,  # Direct server doesn't have manifest endpoint
        "tools_url": "http://localhost:8001/tools",
        "description": "Direct MCP Server with Model Serving"
    },
    "ipfs-accelerate-py": {
        "url": "http://localhost:8000",
        "sse_url": "http://localhost:8000/sse",
        "manifest_url": None,
        "description": "IPFS Accelerate Python MCP Server"
    }
}

SETTINGS_PATH = os.path.expanduser("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
BACKUP_SETTINGS_PATH = os.path.expanduser("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json.bak")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify MCP server connections")
    parser.add_argument("--start-servers", action="store_true", help="Start both MCP servers if not running")
    parser.add_argument("--install-settings", action="store_true", help="Install updated MCP settings")
    parser.add_argument("--verify-connections", action="store_true", help="Verify the connections to all MCP servers")
    parser.add_argument("--test-model-server", action="store_true", help="Test model server functionality")
    parser.add_argument("--test-api-multiplexer", action="store_true", help="Test API multiplexer functionality")
    
    # If no arguments are provided, set verify-connections to true
    args = parser.parse_args()
    if not any(vars(args).values()):
        args.verify_connections = True
    
    return args

def check_server_running(server_name: str) -> bool:
    """Check if a server is running by making an HTTP request."""
    server_info = MCP_SERVERS.get(server_name)
    if not server_info:
        print(f"Unknown server: {server_name}")
        return False
    
    try:
        if server_info["manifest_url"]:
            response = requests.get(server_info["manifest_url"], timeout=2)
            return response.status_code == 200
        elif server_name == "ipfs-direct-mcp" and server_info["tools_url"]:
            response = requests.get(server_info["tools_url"], timeout=2)
            return response.status_code == 200
        else:
            # Try a basic connection to the server URL
            response = requests.get(server_info["url"], timeout=2)
            return response.status_code < 500  # Accept any non-server error
    except Exception as e:
        return False

def start_server(server_name: str) -> bool:
    """Start an MCP server."""
    if server_name == "ipfs-accelerate-mcp":
        try:
            subprocess.Popen(["bash", "restart_mcp_server.sh"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
            time.sleep(3)  # Give it time to start
            return check_server_running(server_name)
        except Exception as e:
            print(f"Error starting IPFS MCP server: {e}")
            return False
    
    elif server_name == "ipfs-direct-mcp":
        try:
            subprocess.Popen(["python", "direct_mcp_server.py", "--port", "8001"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
            time.sleep(2)  # Give it time to start
            return check_server_running(server_name)
        except Exception as e:
            print(f"Error starting Direct MCP server: {e}")
            return False
    
    elif server_name == "ipfs-accelerate-py":
        try:
            subprocess.Popen(["python", "run_ipfs_accelerate_mcp.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
            time.sleep(2)  # Give it time to start
            return check_server_running(server_name)
        except Exception as e:
            print(f"Error starting IPFS Accelerate Python MCP server: {e}")
            return False
    
    else:
        print(f"Unknown server: {server_name}")
        return False

def install_settings():
    """Install the updated MCP settings file."""
    if not os.path.exists("updated_mcp_settings.json"):
        print("Error: updated_mcp_settings.json not found")
        return False
    
    # Create settings directory if it doesn't exist
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    
    # Backup existing settings if they exist
    if os.path.exists(SETTINGS_PATH):
        try:
            shutil.copy2(SETTINGS_PATH, BACKUP_SETTINGS_PATH)
            print(f"Backed up existing settings to {BACKUP_SETTINGS_PATH}")
        except Exception as e:
            print(f"Warning: Failed to backup existing settings: {e}")
    
    # Install new settings
    try:
        shutil.copy2("updated_mcp_settings.json", SETTINGS_PATH)
        print(f"Installed updated MCP settings to {SETTINGS_PATH}")
        return True
    except Exception as e:
        print(f"Error installing settings: {e}")
        return False

def verify_connections():
    """Verify connections to all MCP servers."""
    results = {}
    
    for server_name, server_info in MCP_SERVERS.items():
        running = check_server_running(server_name)
        results[server_name] = {
            "running": running,
            "url": server_info["url"],
            "description": server_info["description"]
        }
    
    # Print results
    print("\n=== MCP Server Connection Status ===\n")
    for server_name, status in results.items():
        status_text = "✅ RUNNING" if status["running"] else "❌ NOT RUNNING"
        print(f"{server_name}: {status_text}")
        print(f"  URL: {status['url']}")
        print(f"  Description: {status['description']}")
        print()
    
    return all(status["running"] for status in results.values())

def test_model_server():
    """Test the model server functionality of the Direct MCP server."""
    server_url = MCP_SERVERS["ipfs-direct-mcp"]["url"]
    
    print("\n=== Testing Model Server Functionality ===\n")
    
    # Step 1: List models
    print("1. Listing available models...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "list_models", "arguments": {}},
            timeout=5
        )
        if response.status_code == 200:
            models = response.json()["result"]["models"]
            print(f"✅ Found {len(models)} models:")
            for model_name, model_info in models.items():
                print(f"  - {model_name}: {model_info['type']}")
        else:
            print(f"❌ Failed to list models: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False
    
    # Step 2: Create endpoint
    print("\n2. Creating model endpoint...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "create_endpoint", 
                "arguments": {
                    "model_name": "bert-base-uncased", 
                    "device": "cpu", 
                    "max_batch_size": 8
                }
            },
            timeout=5
        )
        if response.status_code == 200:
            endpoint = response.json()["result"]
            endpoint_id = endpoint["endpoint_id"]
            print(f"✅ Created endpoint: {endpoint_id}")
        else:
            print(f"❌ Failed to create endpoint: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error creating endpoint: {e}")
        return False
    
    # Step 3: Run inference
    print("\n3. Running inference...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "run_inference", 
                "arguments": {
                    "endpoint_id": endpoint_id, 
                    "inputs": ["Hello world", "Testing embeddings"]
                }
            },
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()["result"]
            print(f"✅ Inference successful: {len(result['embeddings'])} embeddings generated")
            print(f"  - Dimensions: {result['dimensions']}")
        else:
            print(f"❌ Failed to run inference: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error running inference: {e}")
        return False
    
    print("\n✅ Model server functionality verified successfully")
    return True

def test_api_multiplexer():
    """Test the API multiplexer functionality of the Direct MCP server."""
    server_url = MCP_SERVERS["ipfs-direct-mcp"]["url"]
    
    print("\n=== Testing API Multiplexer Functionality ===\n")
    
    # Step 1: Get API keys
    print("1. Getting registered API keys...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_api_keys", "arguments": {}},
            timeout=5
        )
        if response.status_code == 200:
            api_keys = response.json()["result"]
            print(f"✅ Found {api_keys['total_keys']} API keys:")
            for provider in api_keys["providers"]:
                print(f"  - {provider['name']}: {provider['key_count']} keys")
        else:
            print(f"❌ Failed to get API keys: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting API keys: {e}")
        return False
    
    # Step 2: Get multiplexer stats
    print("\n2. Getting multiplexer statistics...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_multiplexer_stats", "arguments": {}},
            timeout=5
        )
        if response.status_code == 200:
            stats = response.json()["result"]
            print(f"✅ Retrieved multiplexer stats:")
            print(f"  - Total requests: {stats['total_requests']}")
            print(f"  - Successful requests: {stats['successful_requests']}")
            print(f"  - Load balancing strategy: {stats['load_balancing']['strategy']}")
        else:
            print(f"❌ Failed to get multiplexer stats: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting multiplexer stats: {e}")
        return False
    
    # Step 3: Simulate API request
    print("\n3. Simulating API request...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "simulate_api_request", 
                "arguments": {
                    "provider": "openai", 
                    "prompt": "Test of the API multiplexer"
                }
            },
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()["result"]
            print(f"✅ API request simulation successful:")
            print(f"  - Provider: {result['provider']}")
            print(f"  - Model: {result['model']}")
            print(f"  - Latency: {result['latency_ms']}ms")
        else:
            print(f"❌ Failed to simulate API request: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error simulating API request: {e}")
        return False
    
    print("\n✅ API multiplexer functionality verified successfully")
    return True

def main():
    """Main function."""
    args = parse_args()
    
    if args.start_servers:
        print("Starting MCP servers...")
        for server_name in MCP_SERVERS:
            if not check_server_running(server_name):
                print(f"Starting {server_name}...")
                if start_server(server_name):
                    print(f"✅ {server_name} started successfully")
                else:
                    print(f"❌ Failed to start {server_name}")
            else:
                print(f"✅ {server_name} is already running")
    
    if args.install_settings:
        print("Installing updated MCP settings...")
        install_settings()
    
    if args.verify_connections:
        verify_connections()
    
    if args.test_model_server:
        test_model_server()
    
    if args.test_api_multiplexer:
        test_api_multiplexer()

if __name__ == "__main__":
    main()
