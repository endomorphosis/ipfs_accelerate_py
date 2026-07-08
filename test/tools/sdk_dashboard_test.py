#!/usr/bin/env python3
"""
SDK Dashboard Test - No Browser Automation

This script tests the SDK Dashboard and JSON-RPC server without browser automation.
It verifies the core functionality by making direct HTTP requests.
"""

import anyio
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
import signal

import requests

class SDKDashboardTester:
    """Test the SDK Dashboard without browser automation."""
    
    def __init__(self):
        """Initialize the tester."""
        self.dashboard_url = "http://localhost:8080"
        self.jsonrpc_url = "http://localhost:8000"
        self.dashboard_process = None
        self.test_results = {
            "dashboard_server": False,
            "jsonrpc_server": False,
            "jsonrpc_methods": {},
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        }
    
    def start_dashboard(self):
        """Start the SDK dashboard application."""
        print("🔧 Starting SDK Dashboard application...")
        
        # Start the dashboard in the background
        self.dashboard_process = subprocess.Popen(
            [sys.executable, "sdk_dashboard_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Wait for servers to start
        print("⏳ Waiting for servers to start...")
        time.sleep(8)
        
        # Check if process is still running
        if self.dashboard_process.poll() is not None:
            stdout, stderr = self.dashboard_process.communicate()
            print(f"❌ Dashboard process failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            raise RuntimeError("Failed to start dashboard")
        
        print("✅ Dashboard application started")
    
    def test_dashboard_server(self):
        """Test that the dashboard server is responding."""
        print("🌐 Testing dashboard server...")
        
        try:
            response = requests.get(f"{self.dashboard_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Dashboard server responding: {data}")
                self.test_results["dashboard_server"] = True
            else:
                print(f"❌ Dashboard server error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Dashboard server connection failed: {e}")
    
    def test_jsonrpc_server(self):
        """Test that the JSON-RPC server is responding."""
        print("🔗 Testing JSON-RPC server...")
        
        try:
            # Test root endpoint
            response = requests.get(f"{self.jsonrpc_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ JSON-RPC server responding with {len(data.get('methods', []))} methods")
                self.test_results["jsonrpc_server"] = True
                return True
            else:
                print(f"❌ JSON-RPC server error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ JSON-RPC server connection failed: {e}")
            return False
    
    def make_jsonrpc_request(self, method, params=None):
        """Make a JSON-RPC request."""
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        response = requests.post(
            f"{self.jsonrpc_url}/jsonrpc",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def test_jsonrpc_methods(self):
        """Test various JSON-RPC methods."""
        print("🔧 Testing JSON-RPC methods...")
        
        methods_to_test = [
            ("get_server_info", {}),
            ("list_methods", {}),
            ("list_models", {}),
            ("get_model_recommendations", {"task_type": "text_generation", "input_type": "text"}),
            ("generate_text", {"prompt": "Hello world", "max_length": 50}),
            ("classify_text", {"text": "I love this product!"}),
            ("generate_embeddings", {"text": "Sample text for embedding"}),
            ("search_models", {"query": "gpt", "limit": 5}),
        ]
        
        for method, params in methods_to_test:
            try:
                print(f"  Testing {method}...")
                result = self.make_jsonrpc_request(method, params)
                
                if "error" in result:
                    print(f"    ❌ {method} returned error: {result['error']}")
                    self.test_results["jsonrpc_methods"][method] = False
                elif "result" in result:
                    print(f"    ✅ {method} succeeded")
                    self.test_results["jsonrpc_methods"][method] = True
                    
                    # Print some details for key methods
                    if method == "get_server_info":
                        server_info = result["result"]
                        print(f"      Server: {server_info.get('name')} v{server_info.get('version')}")
                    elif method == "list_methods":
                        methods_list = result["result"]
                        print(f"      Found {len(methods_list)} available methods")
                    elif method == "generate_text":
                        generated = result["result"]
                        print(f"      Generated: {generated.get('generated_text', '')[:50]}...")
                else:
                    print(f"    ⚠️ {method} returned unexpected format")
                    self.test_results["jsonrpc_methods"][method] = False
                    
            except Exception as e:
                print(f"    ❌ {method} failed: {e}")
                self.test_results["jsonrpc_methods"][method] = False
    
    def test_dashboard_static_files(self):
        """Test that static files are served correctly."""
        print("📁 Testing static file serving...")
        
        static_files = [
            "/static/js/mcp-sdk.js",
            "/static/js/kitchen-sink-sdk.js"
        ]
        
        for file_path in static_files:
            try:
                response = requests.get(f"{self.dashboard_url}{file_path}", timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {file_path} served successfully ({len(response.content)} bytes)")
                else:
                    print(f"  ❌ {file_path} failed: {response.status_code}")
            except Exception as e:
                print(f"  ❌ {file_path} error: {e}")
    
    def test_dashboard_html(self):
        """Test that the main dashboard HTML is served."""
        print("🌐 Testing dashboard HTML...")
        
        try:
            response = requests.get(self.dashboard_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                if "Kitchen Sink AI Model Testing Interface" in content:
                    print("  ✅ Dashboard HTML contains expected title")
                if "mcp-sdk.js" in content:
                    print("  ✅ Dashboard HTML includes MCP SDK")
                if "kitchen-sink-sdk.js" in content:
                    print("  ✅ Dashboard HTML includes Kitchen Sink SDK")
                print(f"  📊 Dashboard HTML served ({len(content)} characters)")
            else:
                print(f"  ❌ Dashboard HTML failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Dashboard HTML error: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        print("🧹 Cleaning up...")
        
        if self.dashboard_process:
            print("  Terminating dashboard process...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("  Force killing dashboard process...")
                self.dashboard_process.kill()
        
        print("✅ Cleanup complete")
    
    def generate_report(self):
        """Generate a test report."""
        print("\n" + "="*60)
        print("📊 SDK DASHBOARD TEST REPORT")
        print("="*60)
        
        # Calculate statistics
        basic_tests = [
            self.test_results["dashboard_server"],
            self.test_results["jsonrpc_server"]
        ]
        
        method_tests = list(self.test_results["jsonrpc_methods"].values())
        all_tests = basic_tests + method_tests
        
        self.test_results["total_tests"] = len(all_tests)
        self.test_results["passed_tests"] = sum(all_tests)
        self.test_results["failed_tests"] = self.test_results["total_tests"] - self.test_results["passed_tests"]
        
        success_rate = (self.test_results["passed_tests"] / self.test_results["total_tests"] * 100) if self.test_results["total_tests"] > 0 else 0
        
        print(f"📈 SUMMARY:")
        print(f"  Total Tests: {self.test_results['total_tests']}")
        print(f"  Passed: {self.test_results['passed_tests']}")
        print(f"  Failed: {self.test_results['failed_tests']}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        print(f"\n🔧 BASIC FUNCTIONALITY:")
        print(f"  Dashboard Server: {'✅ PASS' if self.test_results['dashboard_server'] else '❌ FAIL'}")
        print(f"  JSON-RPC Server: {'✅ PASS' if self.test_results['jsonrpc_server'] else '❌ FAIL'}")
        
        if self.test_results["jsonrpc_methods"]:
            print(f"\n🎯 JSON-RPC METHODS:")
            for method, result in self.test_results["jsonrpc_methods"].items():
                status = '✅ PASS' if result else '❌ FAIL'
                print(f"  {method}: {status}")
        
        print(f"\n🎉 CONCLUSION:")
        if self.test_results["failed_tests"] == 0:
            print("  ALL TESTS PASSED! The SDK Dashboard is fully functional.")
            print("  ✅ JSON-RPC communication working")
            print("  ✅ All inference endpoints responding")
            print("  ✅ Ready for browser-based testing")
        else:
            print(f"  Some tests failed. Please review the {self.test_results['failed_tests']} failed test(s).")
            
        return self.test_results["failed_tests"] == 0
    
    def run_all_tests(self):
        """Run all tests."""
        print("🤖 Starting SDK Dashboard Testing (No Browser)")
        print("="*60)
        
        try:
            # Start the application
            self.start_dashboard()
            
            # Basic connectivity tests
            self.test_dashboard_server()
            
            if self.test_jsonrpc_server():
                # JSON-RPC functionality tests
                self.test_jsonrpc_methods()
            else:
                print("⚠️ Skipping JSON-RPC method tests - server not responding")
            
            # Static file tests
            self.test_dashboard_static_files()
            self.test_dashboard_html()
            
            # Generate report
            return self.generate_report()
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main function."""
    tester = SDKDashboardTester()
    success = tester.run_all_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)