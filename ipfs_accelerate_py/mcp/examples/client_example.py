"""
IPFS Accelerate MCP Client Example

This example demonstrates how to use the MCP client to interact with the IPFS Accelerate MCP Server.
"""

import json
import requests
import sys

def main():
    """
    Main function demonstrating the MCP client
    """
    print("IPFS Accelerate MCP Client Example")
    print("----------------------------------\n")
    
    # Server URL
    base_url = "http://localhost:8000"
    mcp_path = "/mcp"
    
    # Test server connection
    try:
        print("1. Testing server connection...")
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✓ Server is running and documentation is accessible\n")
        else:
            print(f"✗ Server documentation returned status code {response.status_code}\n")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}\n")
        sys.exit(1)
    
    # Get hardware information
    try:
        print("2. Getting hardware information...")
        response = requests.post(
            f"{base_url}{mcp_path}/tool/get_hardware_info",
            json={}
        )
        
        if response.status_code == 200:
            hardware_info = response.json()
            print("✓ Hardware information retrieved successfully")
            print(f"  - CPU: {hardware_info.get('cpu', {}).get('model', 'Unknown')}")
            print(f"  - Memory: {hardware_info.get('memory', {}).get('total', 'Unknown')} GB")
            print(f"  - GPU: {hardware_info.get('gpu', {}).get('name', 'Unknown')}\n")
            
            # Save hardware info to file for reference
            with open("hardware_info.json", "w") as f:
                json.dump(hardware_info, f, indent=2)
                print("  - Hardware information saved to hardware_info.json\n")
        else:
            print(f"✗ Failed to get hardware information: Status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to get hardware information: {e}\n")
    
    # Get supported models
    try:
        print("3. Getting supported models...")
        response = requests.get(
            f"{base_url}{mcp_path}/resource/ipfs_accelerate/supported_models"
        )
        
        if response.status_code == 200:
            model_info = response.json()
            print("✓ Supported models information retrieved successfully")
            print(f"  - Total models: {model_info.get('count', 0)}")
            
            # Print categories
            categories = model_info.get('categories', {})
            print(f"  - Categories: {', '.join(categories.keys())}")
            
            # Print models in each category
            for category_name, category in categories.items():
                print(f"\n  * {category_name} models:")
                models = category.get('models', [])
                for model in models:
                    print(f"    - {model.get('name', 'Unknown')}: {model.get('description', 'No description')}")
            
            # Save model info to file for reference
            with open("supported_models.json", "w") as f:
                json.dump(model_info, f, indent=2)
                print("\n  - Supported models information saved to supported_models.json\n")
        else:
            print(f"✗ Failed to get supported models: Status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to get supported models: {e}\n")
    
    # Test hardware compatibility
    try:
        print("4. Testing hardware compatibility...")
        response = requests.post(
            f"{base_url}{mcp_path}/tool/test_hardware",
            json={}
        )
        
        if response.status_code == 200:
            compatibility = response.json()
            print("✓ Hardware compatibility test completed")
            print(f"  - Overall compatibility: {compatibility.get('overall_compatibility', 'Unknown')}")
            print(f"  - CPU compatibility: {compatibility.get('cpu_compatibility', 'Unknown')}")
            print(f"  - GPU compatibility: {compatibility.get('gpu_compatibility', 'Unknown')}")
            print(f"  - Memory compatibility: {compatibility.get('memory_compatibility', 'Unknown')}\n")
        else:
            print(f"✗ Failed to test hardware compatibility: Status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to test hardware compatibility: {e}\n")
    
    # Get hardware recommendations
    try:
        print("5. Getting hardware recommendations...")
        response = requests.post(
            f"{base_url}{mcp_path}/tool/recommend_hardware",
            json={"model_name": "llama-7b"}  # Model name is required
        )
        
        if response.status_code == 200:
            recommendations = response.json()
            print("✓ Hardware recommendations retrieved successfully")
            print("  Recommendations:")
            for category, rec in recommendations.items():
                if isinstance(rec, dict):
                    print(f"  - {category}:")
                    for key, value in rec.items():
                        print(f"    - {key}: {value}")
                else:
                    print(f"  - {category}: {rec}")
            print()
        else:
            print(f"✗ Failed to get hardware recommendations: Status code {response.status_code}")
            print(f"  Error: {response.text}\n")
    except Exception as e:
        print(f"✗ Failed to get hardware recommendations: {e}\n")
    
    print("Example completed!")

if __name__ == "__main__":
    main()
