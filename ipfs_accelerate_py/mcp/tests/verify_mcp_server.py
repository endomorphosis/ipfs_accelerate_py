"""
IPFS Accelerate MCP Server Verification Script

This script connects to the IPFS Accelerate MCP server, lists all registered
tools and resources, and verifies that they are properly accessible.
"""

import os
import sys
import json
import anyio
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Try to import storage wrapper with comprehensive fallback
try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False
            def get_storage_wrapper(*args, **kwargs):
                return None


class MCPVerifier:
    """Verify IPFS Accelerate MCP Server registration and accessibility"""
    
    def __init__(self):
        """Initialize the verifier"""
        self.server_process = None
        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "ipfs_accelerate_py.mcp.server"],
        )
        self.results = {
            "tools": {},
            "resources": {},
            "prompts": {},
            "summary": {}
        }
        # Initialize storage wrapper
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        
    async def start_server(self):
        """Start the MCP server"""
        print("Starting MCP server...")
        # Start the server in a separate process
        self.server_process = subprocess.Popen(
            [sys.executable, "-m", "ipfs_accelerate_py.mcp.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Wait a moment for the server to initialize
        time.sleep(2)
        print("MCP server started with PID:", self.server_process.pid)
        
    def stop_server(self):
        """Stop the MCP server"""
        if self.server_process is not None:
            print("Stopping MCP server...")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.server_process = None
            print("MCP server stopped")
    
    async def verify(self):
        """Run verification and return the results"""
        try:
            await self.start_server()
            
            # Connect to the server
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()
                    print("Connected to MCP server")
                    
                    # Verify tools
                    await self.verify_tools(session)
                    
                    # Verify resources
                    await self.verify_resources(session)
                    
                    # Verify prompts
                    await self.verify_prompts(session)
                    
                    # Prepare summary
                    self.prepare_summary()
            
            return self.results
        
        finally:
            # Always stop the server when we're done
            self.stop_server()
    
    async def verify_tools(self, session):
        """Verify that all expected tools are registered"""
        print("\nVerifying Tools:")
        try:
            tools = await session.list_tools()
            tool_names = sorted([tool.name for tool in tools])
            
            print(f"Found {len(tools)} tools:")
            for i, tool_name in enumerate(tool_names, 1):
                print(f"  {i}. {tool_name}")
            
            # Categorize tools
            expected_tools = {
                # Inference tools
                "run_inference": "Inference",
                "batch_inference": "Inference",
                "get_model_types": "Inference",
                
                # Endpoint tools
                "list_endpoints": "Endpoint",
                "add_endpoint": "Endpoint",
                "remove_endpoint": "Endpoint",
                "test_endpoint": "Endpoint",
                
                # Hardware tools
                "test_hardware": "Hardware",
                "get_batch_sizes": "Hardware",
                "get_hardware_info": "Hardware",
                
                # Status tools
                "get_system_status": "Status",
                "monitor_performance": "Status",
                "get_active_models": "Status"
            }
            
            # Check which expected tools are present
            categories = {}
            for tool_name in tool_names:
                category = expected_tools.get(tool_name, "Other")
                if category not in categories:
                    categories[category] = []
                categories[category].append(tool_name)
            
            # Calculate metrics
            found_expected_tools = [t for t in tool_names if t in expected_tools]
            missing_tools = [t for t in expected_tools if t not in tool_names]
            unexpected_tools = [t for t in tool_names if t not in expected_tools]
            
            # Store results
            self.results["tools"] = {
                "total_count": len(tools),
                "found_expected_count": len(found_expected_tools),
                "missing_count": len(missing_tools),
                "unexpected_count": len(unexpected_tools),
                "by_category": categories,
                "found_expected_tools": found_expected_tools,
                "missing_tools": missing_tools,
                "unexpected_tools": unexpected_tools,
                "all_tools": tool_names
            }
            
            # Print summary
            print(f"\nTools Summary:")
            print(f"  Total tools: {len(tools)}")
            print(f"  Expected tools found: {len(found_expected_tools)}/{len(expected_tools)}")
            if missing_tools:
                print(f"  Missing tools: {', '.join(missing_tools)}")
            if unexpected_tools:
                print(f"  Unexpected tools: {', '.join(unexpected_tools)}")
            
            print("\nTools by Category:")
            for category, tools_list in categories.items():
                print(f"  {category}: {len(tools_list)} tools")
                for tool in tools_list:
                    print(f"    - {tool}")
            
        except Exception as e:
            print(f"Error verifying tools: {e}")
            self.results["tools"] = {"error": str(e)}
    
    async def verify_resources(self, session):
        """Verify that all expected resources are registered"""
        print("\nVerifying Resources:")
        try:
            resources = await session.list_resources()
            resource_uris = sorted([resource.uri for resource in resources])
            
            print(f"Found {len(resources)} resources:")
            for i, uri in enumerate(resource_uris, 1):
                print(f"  {i}. {uri}")
            
            # Expected resources
            expected_resources = [
                "ipfs://config",
                "ipfs://endpoints",
                "ipfs://models"
            ]
            
            # Check which expected resources are present
            found_expected_resources = [r for r in resource_uris if r in expected_resources]
            missing_resources = [r for r in expected_resources if r not in resource_uris]
            pattern_resources = [r for r in resource_uris if "{" in r]
            
            # Store results
            self.results["resources"] = {
                "total_count": len(resources),
                "found_expected_count": len(found_expected_resources),
                "missing_count": len(missing_resources),
                "pattern_resources": pattern_resources,
                "found_expected_resources": found_expected_resources,
                "missing_resources": missing_resources,
                "all_resources": resource_uris
            }
            
            # Print summary
            print(f"\nResources Summary:")
            print(f"  Total resources: {len(resources)}")
            print(f"  Expected resources found: {len(found_expected_resources)}/{len(expected_resources)}")
            if missing_resources:
                print(f"  Missing resources: {', '.join(missing_resources)}")
            if pattern_resources:
                print(f"  Pattern resources: {', '.join(pattern_resources)}")
            
            # Test accessing a resource
            if "ipfs://config" in resource_uris:
                print("\nTesting resource access for ipfs://config:")
                try:
                    content, mime_type = await session.read_resource("ipfs://config")
                    config = json.loads(content)
                    config_json = json.dumps(config, indent=2)
                    print(f"  Successfully accessed ipfs://config (mime-type: {mime_type})")
                    print(f"  Config keys: {', '.join(config.keys()) if isinstance(config, dict) else 'N/A'}")
                    
                    # Try to save config to distributed storage
                    if self._storage:
                        try:
                            cid = self._storage.write_file(config_json, "mcp_server_config.json", pin=False)
                            print(f"  Config saved to distributed storage: {cid}")
                        except Exception as e:
                            print(f"  Could not save to distributed storage: {e}")
                    
                    self.results["resources"]["config_access_test"] = "success"
                except Exception as e:
                    print(f"  Error accessing ipfs://config: {e}")
                    self.results["resources"]["config_access_test"] = str(e)
            
        except Exception as e:
            print(f"Error verifying resources: {e}")
            self.results["resources"] = {"error": str(e)}
    
    async def verify_prompts(self, session):
        """Verify that prompts are registered"""
        print("\nVerifying Prompts:")
        try:
            prompts = await session.list_prompts()
            prompt_names = sorted([prompt.name for prompt in prompts])
            
            print(f"Found {len(prompts)} prompts:")
            for i, name in enumerate(prompt_names, 1):
                print(f"  {i}. {name}")
            
            # Expected prompts
            expected_prompts = [
                "run_inference_example",
                "system_status_example",
                "batch_inference_example",
                "add_endpoint_example",
                "model_comparison_workflow"
            ]
            
            # Check which expected prompts are present
            found_expected_prompts = [p for p in prompt_names if p in expected_prompts]
            missing_prompts = [p for p in expected_prompts if p not in prompt_names]
            unexpected_prompts = [p for p in prompt_names if p not in expected_prompts]
            
            # Store results
            self.results["prompts"] = {
                "total_count": len(prompts),
                "found_expected_count": len(found_expected_prompts),
                "missing_count": len(missing_prompts),
                "unexpected_count": len(unexpected_prompts),
                "found_expected_prompts": found_expected_prompts,
                "missing_prompts": missing_prompts,
                "unexpected_prompts": unexpected_prompts,
                "all_prompts": prompt_names
            }
            
            # Print summary
            print(f"\nPrompts Summary:")
            print(f"  Total prompts: {len(prompts)}")
            print(f"  Expected prompts found: {len(found_expected_prompts)}/{len(expected_prompts)}")
            if missing_prompts:
                print(f"  Missing prompts: {', '.join(missing_prompts)}")
            if unexpected_prompts:
                print(f"  Unexpected prompts: {', '.join(unexpected_prompts)}")
            
        except Exception as e:
            print(f"Error verifying prompts: {e}")
            self.results["prompts"] = {"error": str(e)}
    
    def prepare_summary(self):
        """Prepare summary metrics"""
        # Count total expected items
        total_expected = (
            len(self.results["tools"].get("found_expected_tools", [])) +
            len(self.results["tools"].get("missing_tools", [])) +
            len(self.results["resources"].get("found_expected_resources", [])) +
            len(self.results["resources"].get("missing_resources", [])) +
            len(self.results["prompts"].get("found_expected_prompts", [])) +
            len(self.results["prompts"].get("missing_prompts", []))
        )
        
        # Count total found expected items
        total_found = (
            len(self.results["tools"].get("found_expected_tools", [])) +
            len(self.results["resources"].get("found_expected_resources", [])) +
            len(self.results["prompts"].get("found_expected_prompts", []))
        )
        
        # Calculate overall completion percentage
        completion_percentage = round((total_found / total_expected) * 100, 1) if total_expected > 0 else 0
        
        # Prepare summary
        self.results["summary"] = {
            "total_expected_items": total_expected,
            "total_found_items": total_found,
            "completion_percentage": completion_percentage,
            "tools_completion": f"{len(self.results['tools'].get('found_expected_tools', []))}/{len(self.results['tools'].get('found_expected_tools', [])) + len(self.results['tools'].get('missing_tools', []))}",
            "resources_completion": f"{len(self.results['resources'].get('found_expected_resources', []))}/{len(self.results['resources'].get('found_expected_resources', [])) + len(self.results['resources'].get('missing_resources', []))}",
            "prompts_completion": f"{len(self.results['prompts'].get('found_expected_prompts', []))}/{len(self.results['prompts'].get('found_expected_prompts', [])) + len(self.results['prompts'].get('missing_prompts', []))}"
        }


async def main():
    """Run the verification and print results"""
    # Make sure the output directories exist
    os.makedirs("ipfs_accelerate_py/mcp/tests", exist_ok=True)
    
    print("=" * 80)
    print("IPFS Accelerate MCP Server Verification")
    print("=" * 80)
    
    # Run verification
    verifier = MCPVerifier()
    results = await verifier.verify()
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nOverall Completion: {summary['completion_percentage']}% ({summary['total_found_items']}/{summary['total_expected_items']} items)")
        print(f"- Tools: {summary['tools_completion']}")
        print(f"- Resources: {summary['resources_completion']}")
        print(f"- Prompts: {summary['prompts_completion']}")
    
    # Save results to file
    results_path = "ipfs_accelerate_py/mcp/tests/verification_results.json"
    results_json = json.dumps(results, indent=2)
    
    # Try distributed storage first
    storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
    if storage:
        try:
            cid = storage.write_file(results_json, results_path, pin=True)
            print(f"\nDetailed verification results saved to distributed storage: {cid}")
        except Exception as e:
            print(f"Distributed storage failed: {e}, using local storage")
            with open(results_path, "w") as f:
                f.write(results_json)
            print(f"\nDetailed verification results saved to: {results_path}")
    else:
        with open(results_path, "w") as f:
            f.write(results_json)
        print(f"\nDetailed verification results saved to: {results_path}")
    
    # Return overall status
    if "summary" in results and results["summary"]["completion_percentage"] >= 100:
        print("\n✅ ALL EXPECTED COMPONENTS VERIFIED SUCCESSFULLY")
        return 0
    else:
        print("\n❌ SOME EXPECTED COMPONENTS ARE MISSING")
        return 1


if __name__ == "__main__":
    exit_code = anyio.run(main())
    sys.exit(exit_code)
