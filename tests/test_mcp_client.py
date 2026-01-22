"""
IPFS Accelerate MCP Client Test

This script tests the MCP client by connecting to the MCP server and using tools.
"""

import json
import requests
import sys

def main():
    """
    Main function to test the MCP client
    """
    print("IPFS Accelerate MCP Client Test")
    print("-------------------------------\n")
    
    # Server URL
    base_url = "http://localhost:8000"
    mcp_path = "/mcp"
    
    # Test server connection
    try:
        print("Testing server connection...")
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✓ Server is running and documentation is accessible\n")
        else:
            print(f"✗ Server documentation returned status code {response.status_code}\n")
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}\n")
        sys.exit(1)
    
    # Test hardware tool
    try:
        print("Testing 'get_hardware_info' tool...")
        response = requests.post(
            f"{base_url}{mcp_path}/tool/get_hardware_info",
            json={}
        )
        
        if response.status_code == 200:
            hardware_info = response.json()
            print("✓ Tool executed successfully")
            print(f"  - CPU: {hardware_info.get('cpu', {}).get('model', 'Unknown')}")
            print(f"  - Memory: {hardware_info.get('memory', {}).get('total', 'Unknown')} GB")
            print(f"  - GPU: {hardware_info.get('gpu', {}).get('name', 'Unknown')}")
            print()
        else:
            print(f"✗ Tool returned status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to execute tool: {e}\n")
    
    # Test model info resource
    try:
        print("Testing 'ipfs_accelerate/supported_models' resource...")
        response = requests.get(
            f"{base_url}{mcp_path}/resource/ipfs_accelerate/supported_models"
        )
        
        if response.status_code == 200:
            model_info = response.json()
            print("✓ Resource accessed successfully")
            print(f"  - Total models: {model_info.get('count', 0)}")
            
            # Print categories
            categories = model_info.get('categories', {})
            print(f"  - Categories: {', '.join(categories.keys())}")
            
            # Print first model in each category
            for category_name, category in categories.items():
                models = category.get('models', [])
                if models:
                    model = models[0]
                    print(f"    * {category_name}: {model.get('name', 'Unknown')} - {model.get('description', 'No description')}")
            print()
        else:
            print(f"✗ Resource returned status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to access resource: {e}\n")
    
    print("Test completed!")

if __name__ == "__main__":
    main()
