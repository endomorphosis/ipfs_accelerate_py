#!/usr/bin/env python3
"""
Focused test for the get_hardware_info MCP tool.
This script specifically tests the hardware_info tool that has been problematic.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_test")

class HardwareToolTester:
    """Test class for the hardware_info tool."""
    
    def __init__(self, host="127.0.0.1", port=8002, timeout=5):
        """Initialize with server connection details."""
        self.base_url = f"http://{host}:{port}"
        self.tools_url = f"{self.base_url}/tools"
        self.hardware_tool_url = f"{self.tools_url}/get_hardware_info/invoke"
        self.timeout = timeout
    
    def check_server_connection(self):
        """Check if the server is up and running."""
        try:
            response = requests.get(self.base_url, timeout=self.timeout)
            if response.status_code == 200:
                logger.info(f"Server connection successful: {response.status_code}")
                return True
            else:
                logger.warning(f"Server returned status code: {response.status_code}")
                return False
        except requests.ConnectionError:
            logger.error(f"Failed to connect to server at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            return False
    
    def list_tools(self):
        """Get the list of available tools."""
        try:
            response = requests.get(self.tools_url, timeout=self.timeout)
            if response.status_code == 200:
                tools = response.json()
                logger.info(f"Found {len(tools)} tools: {', '.join(tools)}")
                return tools
            else:
                logger.warning(f"Failed to list tools: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    def test_hardware_tool(self):
        """Test the get_hardware_info tool specifically."""
        try:
            logger.info(f"Testing get_hardware_info at {self.hardware_tool_url}")
            response = requests.post(self.hardware_tool_url, json={}, timeout=self.timeout)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"get_hardware_info test succeeded")
                
                # Verify the structure of the result
                if isinstance(result, dict) and "system" in result:
                    logger.info("Hardware info has expected structure with 'system' key")
                    return True, result
                else:
                    logger.warning("Hardware info does not have expected structure")
                    return False, result
            else:
                logger.warning(f"get_hardware_info test failed with status {response.status_code}: {response.text}")
                return False, None
        except Exception as e:
            logger.error(f"Error testing get_hardware_info: {e}")
            return False, None

def main():
    """Main function to test the hardware_info tool."""
    parser = argparse.ArgumentParser(description='Test the get_hardware_info tool')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8002, help='Server port')
    parser.add_argument('--timeout', type=int, default=5, help='Request timeout')
    parser.add_argument('--output', type=str, default='hardware_info_test.json', help='Output file for results')
    args = parser.parse_args()
    
    try:
        # Create tester instance
        tester = HardwareToolTester(args.host, args.port, args.timeout)
        
        # Check if the server is running
        if not tester.check_server_connection():
            logger.error("Cannot connect to the MCP server. Please start it first.")
            return 1
        
        # List available tools
        tools = tester.list_tools()
        if "get_hardware_info" not in tools:
            logger.error("get_hardware_info tool not available on the server")
            return 1
        
        # Test the hardware_info tool
        success, result = tester.test_hardware_tool()
        
        # Save the result to a file
        with open(args.output, 'w') as f:
            json.dump(result if result else {"error": "Failed to get hardware info"}, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Print the result
        print("\n" + "=" * 50)
        print("HARDWARE INFO TOOL TEST RESULT:")
        print("=" * 50)
        print(f"Status: {'SUCCESS' if success else 'FAILED'}")
        if result:
            print("\nHardware Information:")
            print(json.dumps(result, indent=2))
        print("=" * 50)
        
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
