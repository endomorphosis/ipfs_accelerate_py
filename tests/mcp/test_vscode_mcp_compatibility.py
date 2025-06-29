#!/usr/bin/env python3
"""
VS Code MCP Integration Test
Tests all MCP tools for compatibility with VS Code and proper functionality.
"""

import json
import sys
import time
import subprocess
import requests
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VSCodeMCPTester:
    """Test MCP server compatibility with VS Code"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8002):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.test_results = {}
        
    def start_server_if_needed(self) -> bool:
        """Start MCP server if not already running"""
        try:
            # Check if server is already running
            response = requests.get(f"{self.base_url}/mcp/manifest", timeout=5)
            if response.status_code == 200:
                logger.info("MCP server is already running")
                return True
        except requests.exceptions.RequestException:
            pass
            
        # Start the server
        logger.info("Starting MCP server...")
        try:
            self.server_process = subprocess.Popen(
                ["python3", "final_mcp_server.py", "--host", self.host, "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Wait for server to start
            for _ in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"{self.base_url}/mcp/manifest", timeout=2)
                    if response.status_code == 200:
                        logger.info("MCP server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    
            logger.error("Server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    def test_manifest_endpoint(self) -> bool:
        """Test the manifest endpoint"""
        try:
            response = requests.get(f"{self.base_url}/mcp/manifest", timeout=10)
            if response.status_code == 200:
                manifest = response.json()
                self.test_results['manifest'] = {
                    'success': True,
                    'tools_count': len(manifest.get('tools', [])),
                    'tools': manifest.get('tools', []),
                    'server': manifest.get('server'),
                    'version': manifest.get('version')
                }
                logger.info(f"✅ Manifest: {len(manifest.get('tools', []))} tools available")
                return True
            else:
                self.test_results['manifest'] = {
                    'success': False, 
                    'error': f"HTTP {response.status_code}"
                }
                logger.error(f"❌ Manifest endpoint returned {response.status_code}")
                return False
        except Exception as e:
            self.test_results['manifest'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Manifest endpoint error: {e}")
            return False
    
    def test_tool_execution(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Test individual tool execution via JSON-RPC"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": params,
                "id": 1
            }
            
            response = requests.post(
                f"{self.base_url}/jsonrpc",
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    self.test_results[f'tool_{tool_name}'] = {
                        'success': True,
                        'result': result.get('result')
                    }
                    logger.info(f"✅ Tool {tool_name}: Working")
                    return True
                else:
                    self.test_results[f'tool_{tool_name}'] = {
                        'success': False,
                        'error': result.get('error')
                    }
                    logger.warning(f"⚠️ Tool {tool_name}: Error - {result.get('error')}")
                    return False
            else:
                self.test_results[f'tool_{tool_name}'] = {
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                }
                logger.error(f"❌ Tool {tool_name}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.test_results[f'tool_{tool_name}'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Tool {tool_name}: Exception - {e}")
            return False
    
    def test_all_core_tools(self) -> int:
        """Test all core tools with appropriate parameters"""
        tools_to_test = [
            ('health_check', {}),
            ('get_hardware_info', {}),
            ('ipfs_add_file', {'path': '/tmp/test.txt'}),
            ('ipfs_cat', {'cid': 'QmTest'}),
            ('ipfs_get', {'cid': 'QmTest', 'output_path': '/tmp/output.txt'}),
            ('ipfs_files_write', {'path': '/test.txt', 'content': 'test content'}),
            ('ipfs_files_read', {'path': '/test.txt'}),
            ('list_models', {}),
            ('create_endpoint', {'model_name': 'test-model'}),
            ('run_inference', {'endpoint_id': 'test', 'inputs': ['test input']}),
            ('process_data', {'model_name': 'test-model', 'input_data': 'test'}),
            ('init_endpoints', {'models': ['test-model']}),
            ('vfs_list', {'path': '/'}),
            ('create_storage', {'name': 'test-storage', 'size': 1.0})
        ]
        
        successful_tools = 0
        
        logger.info("Testing all core tools...")
        for tool_name, params in tools_to_test:
            if self.test_tool_execution(tool_name, params):
                successful_tools += 1
        
        logger.info(f"Tool test results: {successful_tools}/{len(tools_to_test)} tools working")
        return successful_tools
    
    def test_vscode_compatibility(self) -> bool:
        """Test VS Code specific compatibility features"""
        compatibility_checks = {}
        
        # Check if .vscode directory exists
        vscode_dir = Path('.vscode')
        compatibility_checks['vscode_config'] = vscode_dir.exists()
        
        # Check for tasks.json
        tasks_file = vscode_dir / 'tasks.json'
        compatibility_checks['tasks_config'] = tasks_file.exists()
        
        # Check for launch.json
        launch_file = vscode_dir / 'launch.json'
        compatibility_checks['launch_config'] = launch_file.exists()
        
        # Check for settings.json
        settings_file = vscode_dir / 'settings.json'
        compatibility_checks['settings_config'] = settings_file.exists()
        
        # Test if tasks.json has MCP-related tasks
        if tasks_file.exists():
            try:
                with open(tasks_file) as f:
                    tasks = json.load(f)
                mcp_tasks = [task for task in tasks.get('tasks', []) 
                           if 'mcp' in task.get('label', '').lower()]
                compatibility_checks['mcp_tasks_count'] = len(mcp_tasks)
            except Exception as e:
                compatibility_checks['tasks_parse_error'] = str(e)
        
        # Test if launch.json has MCP debug configs
        if launch_file.exists():
            try:
                with open(launch_file) as f:
                    launch = json.load(f)
                mcp_configs = [config for config in launch.get('configurations', [])
                             if 'mcp' in config.get('name', '').lower()]
                compatibility_checks['mcp_debug_configs'] = len(mcp_configs)
            except Exception as e:
                compatibility_checks['launch_parse_error'] = str(e)
        
        self.test_results['vscode_compatibility'] = compatibility_checks
        
        # Summary
        total_checks = 4  # vscode_config, tasks_config, launch_config, settings_config
        passed_checks = sum(1 for key in ['vscode_config', 'tasks_config', 'launch_config', 'settings_config'] 
                          if compatibility_checks.get(key, False))
        
        logger.info(f"VS Code compatibility: {passed_checks}/{total_checks} checks passed")
        return passed_checks >= 3  # At least 3 out of 4 should pass
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        manifest_success = self.test_results.get('manifest', {}).get('success', False)
        tool_results = {k: v for k, v in self.test_results.items() if k.startswith('tool_')}
        successful_tools = sum(1 for result in tool_results.values() if result.get('success', False))
        total_tools = len(tool_results)
        
        vscode_compat = self.test_results.get('vscode_compatibility', {})
        
        report = {
            'timestamp': time.time(),
            'server': {
                'host': self.host,
                'port': self.port,
                'reachable': manifest_success
            },
            'manifest_test': self.test_results.get('manifest', {}),
            'tool_tests': {
                'total': total_tools,
                'successful': successful_tools,
                'success_rate': successful_tools / total_tools if total_tools > 0 else 0,
                'details': tool_results
            },
            'vscode_compatibility': vscode_compat,
            'overall_status': {
                'server_working': manifest_success,
                'tools_working': successful_tools >= total_tools * 0.8,  # 80% success rate
                'vscode_ready': sum(1 for key in ['vscode_config', 'tasks_config', 'launch_config'] 
                                  if vscode_compat.get(key, False)) >= 2
            }
        }
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                logger.info("Server process terminated")
            except Exception as e:
                logger.warning(f"Error terminating server: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test MCP server VS Code compatibility")
    parser.add_argument("--host", default="127.0.0.1", help="MCP server host")
    parser.add_argument("--port", type=int, default=8002, help="MCP server port")
    parser.add_argument("--output", default="vscode_mcp_test_results.json", help="Output file for test results")
    parser.add_argument("--start-server", action="store_true", help="Start server if not running")
    
    args = parser.parse_args()
    
    tester = VSCodeMCPTester(args.host, args.port)
    
    try:
        # Start server if requested
        if args.start_server:
            if not tester.start_server_if_needed():
                logger.error("Failed to start server")
                return 1
        
        # Run tests
        logger.info("Starting VS Code MCP compatibility tests...")
        
        # Test manifest
        if not tester.test_manifest_endpoint():
            logger.error("Manifest test failed - server may not be running")
            return 1
        
        # Test VS Code compatibility
        tester.test_vscode_compatibility()
        
        # Test all tools
        successful_tools = tester.test_all_core_tools()
        
        # Generate report
        report = tester.generate_report()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("VS CODE MCP COMPATIBILITY TEST SUMMARY")
        print("="*60)
        print(f"Server: {args.host}:{args.port}")
        print(f"Reachable: {'✅ Yes' if report['server']['reachable'] else '❌ No'}")
        print(f"Tools Working: {report['tool_tests']['successful']}/{report['tool_tests']['total']} ({report['tool_tests']['success_rate']:.1%})")
        print(f"VS Code Ready: {'✅ Yes' if report['overall_status']['vscode_ready'] else '❌ No'}")
        
        vscode_compat = report['vscode_compatibility']
        print(f"\nVS Code Configuration:")
        print(f"  .vscode directory: {'✅' if vscode_compat.get('vscode_config') else '❌'}")
        print(f"  tasks.json: {'✅' if vscode_compat.get('tasks_config') else '❌'}")
        print(f"  launch.json: {'✅' if vscode_compat.get('launch_config') else '❌'}")
        print(f"  settings.json: {'✅' if vscode_compat.get('settings_config') else '❌'}")
        
        if vscode_compat.get('mcp_tasks_count', 0) > 0:
            print(f"  MCP Tasks: {vscode_compat['mcp_tasks_count']}")
        if vscode_compat.get('mcp_debug_configs', 0) > 0:
            print(f"  MCP Debug Configs: {vscode_compat['mcp_debug_configs']}")
        
        print(f"\nDetailed results saved to: {args.output}")
        print("="*60)
        
        # Return appropriate exit code
        overall_success = (
            report['overall_status']['server_working'] and
            report['overall_status']['tools_working'] and
            report['overall_status']['vscode_ready']
        )
        
        return 0 if overall_success else 1
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())
