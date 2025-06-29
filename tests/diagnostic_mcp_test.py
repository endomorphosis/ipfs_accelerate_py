#!/usr/bin/env python3
"""
Diagnostic MCP Test Script

This script diagnoses and fixes common issues with the IPFS Accelerate MCP server.
It performs a series of tests and attempts automatic fixes where possible.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import platform
import socket
import requests
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diagnostic_mcp_test.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("diagnostic-mcp-test")

# Constants
SERVER_PORT = 8002
SERVER_HOST = "localhost"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MCP_URL = f"{SERVER_URL}/mcp"
SERVER_SCRIPT = "final_mcp_server.py"
TOOL_REGISTRATION_SCRIPT = "fix_mcp_tool_registration.py"

# Global state
server_process = None
diagnostics = {
    "system_info": {},
    "file_checks": {},
    "server_tests": {},
    "tool_tests": {},
    "fixes_applied": []
}

def check_system_info() -> Dict[str, Any]:
    """Collect system information"""
    logger.info("Collecting system information...")
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "username": os.getlogin()
    }
    
    # Check for required packages
    try:
        import aiohttp
        info["aiohttp_version"] = aiohttp.__version__
    except ImportError:
        info["aiohttp_version"] = "Not installed"
    
    try:
        import fastapi
        info["fastapi_version"] = fastapi.__version__
    except ImportError:
        info["fastapi_version"] = "Not installed"
    
    try:
        import uvicorn
        info["uvicorn_version"] = uvicorn.__version__
    except ImportError:
        info["uvicorn_version"] = "Not installed"
    
    return info

def check_files() -> Dict[str, Any]:
    """Check required files"""
    logger.info("Checking required files...")
    
    files = {
        "server_script": os.path.isfile(SERVER_SCRIPT),
        "tool_registration": os.path.isfile(TOOL_REGISTRATION_SCRIPT),
        "requirements": os.path.isfile("requirements.txt")
    }
    
    # Also check permissions
    if files["server_script"]:
        files["server_script_executable"] = os.access(SERVER_SCRIPT, os.X_OK)
    
    return files

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def try_connect_to_server() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Try to connect to the MCP server"""
    try:
        response = requests.get(f"{MCP_URL}/manifest", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        logger.debug(f"Failed to connect to server: {e}")
        return False, None

def start_server(debug: bool = False) -> subprocess.Popen:
    """Start the MCP server"""
    global server_process
    
    logger.info(f"Starting MCP server: {SERVER_SCRIPT}")
    
    cmd = [
        sys.executable,
        SERVER_SCRIPT,
        "--host", "0.0.0.0",
        "--port", str(SERVER_PORT)
    ]
    
    if debug:
        cmd.append("--debug")
    
    # Create log file for server output
    log_file = open("server_diagnostic.log", "w")
    
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    server_process = process
    return process

def stop_server():
    """Stop the MCP server"""
    global server_process
    
    if server_process:
        logger.info("Stopping MCP server...")
        server_process.send_signal(signal.SIGTERM)
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_process = None

def check_server() -> Dict[str, Any]:
    """Check if the server works correctly"""
    if is_port_in_use(SERVER_PORT):
        logger.warning(f"Port {SERVER_PORT} is already in use. Checking if it's the MCP server...")
        
        # Check if it's actually our MCP server
        success, manifest = try_connect_to_server()
        if success:
            logger.info("Found running MCP server")
            return {
                "status": "already_running",
                "manifest": manifest
            }
        else:
            logger.error(f"Port {SERVER_PORT} is in use by another application")
            return {
                "status": "port_in_use_by_other",
                "manifest": None
            }
    
    # Start our own server
    logger.info("No MCP server running. Starting new server...")
    start_server(debug=True)
    
    # Wait for it to start
    time.sleep(2)
    
    for i in range(5):
        success, manifest = try_connect_to_server()
        if success:
            logger.info("MCP server started successfully")
            return {
                "status": "started",
                "manifest": manifest
            }
        time.sleep(2)
    
    logger.error("Failed to start MCP server")
    return {
        "status": "failed_to_start",
        "manifest": None
    }

def fix_port_in_use():
    """Try to fix port in use issue"""
    logger.info("Attempting to fix port in use issue...")
    
    # Find the process using the port
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                f"netstat -ano | findstr {SERVER_PORT}",
                shell=True, text=True, capture_output=True
            )
        else:
            result = subprocess.run(
                f"lsof -i:{SERVER_PORT} | grep LISTEN",
                shell=True, text=True, capture_output=True
            )
            
        logger.info(f"Process using port {SERVER_PORT}: {result.stdout}")
        
        # Try to kill the process (this is dangerous and should be done with caution)
        # In a real scenario, you might want to ask the user first
        if "final_mcp_server" in result.stdout or "python" in result.stdout:
            if platform.system() == "Windows":
                pid = result.stdout.strip().split()[-1]
                subprocess.run(f"taskkill /PID {pid} /F", shell=True)
            else:
                pids = []
                for line in result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) > 1:
                        pids.append(parts[1])
                
                for pid in pids:
                    try:
                        subprocess.run(f"kill -9 {pid}", shell=True)
                        logger.info(f"Killed process with PID {pid}")
                    except Exception as e:
                        logger.error(f"Failed to kill process {pid}: {e}")
            
            # Wait a moment
            time.sleep(2)
            
            return not is_port_in_use(SERVER_PORT)
    except Exception as e:
        logger.error(f"Error trying to fix port in use: {e}")
    
    return False

def fix_server_script_issues() -> bool:
    """Fix common issues with the server script"""
    if not os.path.isfile(SERVER_SCRIPT):
        logger.error(f"Server script not found: {SERVER_SCRIPT}")
        return False
    
    # Make script executable
    if not os.access(SERVER_SCRIPT, os.X_OK):
        logger.info(f"Making {SERVER_SCRIPT} executable")
        try:
            os.chmod(SERVER_SCRIPT, os.stat(SERVER_SCRIPT).st_mode | 0o111)
            diagnostics["fixes_applied"].append(f"Made {SERVER_SCRIPT} executable")
            return True
        except Exception as e:
            logger.error(f"Failed to make script executable: {e}")
            return False
    
    return True

def run_tool_registration_fix() -> bool:
    """Run the tool registration fix script"""
    logger.info("Running tool registration fix script...")
    try:
        result = subprocess.run(
            [sys.executable, TOOL_REGISTRATION_SCRIPT, "--autofix"],
            check=True,
            text=True,
            capture_output=True
        )
        logger.info("Tool registration fix completed successfully")
        diagnostics["fixes_applied"].append("Applied tool registration fixes")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tool registration fix failed: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Error running tool registration fix: {e}")
        return False

def test_server_endpoints() -> Dict[str, Any]:
    """Test various server endpoints"""
    endpoints = {
        "manifest": f"{MCP_URL}/manifest",
        "healthcheck": f"{SERVER_URL}/health",
        "tools": f"{MCP_URL}/tools"
    }
    
    results = {}
    
    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=3)
            results[name] = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
            
            if response.status_code == 200:
                try:
                    results[name]["data"] = response.json()
                except:
                    results[name]["data"] = "Not JSON"
        except Exception as e:
            results[name] = {
                "status_code": None,
                "success": False,
                "error": str(e)
            }
    
    return results

def save_diagnostics():
    """Save the diagnostic information to a file"""
    with open("mcp_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='MCP Server Diagnostic Tool')
    parser.add_argument('--autofix', action='store_true', help='Automatically apply fixes')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    print("="*80)
    print("IPFS Accelerate MCP Server Diagnostic Tool")
    print("="*80)
    
    try:
        # Gather system information
        diagnostics["system_info"] = check_system_info()
        
        # Check required files
        diagnostics["file_checks"] = check_files()
        
        # Check if server script is executable
        if not diagnostics["file_checks"].get("server_script_executable", False):
            if args.autofix:
                fix_server_script_issues()
            else:
                logger.warning(f"Server script {SERVER_SCRIPT} is not executable. Use --autofix to fix this.")
        
        # Check for port conflicts
        if is_port_in_use(SERVER_PORT):
            logger.warning(f"Port {SERVER_PORT} is already in use")
            if args.autofix:
                fix_port_in_use()
        
        # Run the tool registration fix if needed
        if args.autofix and diagnostics["file_checks"].get("tool_registration", False):
            run_tool_registration_fix()
        
        # Save pre-startup diagnostics
        with open("pre_startup_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        
        # Check the server
        diagnostics["server_tests"] = check_server()
        
        # Test endpoints if server is running
        if diagnostics["server_tests"]["status"] in ("already_running", "started"):
            diagnostics["endpoint_tests"] = test_server_endpoints()
        
        # Save final diagnostics
        save_diagnostics()
        
        # Print summary
        print("\nDiagnostic Summary:")
        print("=" * 40)
        print(f"System: {diagnostics['system_info']['platform']}")
        print(f"Python: {diagnostics['system_info']['python_version']}")
        
        print("\nFile Checks:")
        for name, exists in diagnostics["file_checks"].items():
            status = "✅" if exists else "❌"
            print(f"{status} {name}")
        
        print("\nServer Status:")
        status_icons = {
            "already_running": "✅",
            "started": "✅",
            "port_in_use_by_other": "❌",
            "failed_to_start": "❌"
        }
        status = diagnostics["server_tests"]["status"]
        icon = status_icons.get(status, "❓")
        print(f"{icon} {status}")
        
        if "endpoint_tests" in diagnostics:
            print("\nEndpoint Tests:")
            for name, result in diagnostics["endpoint_tests"].items():
                status = "✅" if result["success"] else "❌"
                print(f"{status} {name}")
        
        print("\nFixes Applied:")
        if diagnostics["fixes_applied"]:
            for fix in diagnostics["fixes_applied"]:
                print(f"✅ {fix}")
        else:
            print("None")
        
        print("\nDetailed results saved to: mcp_diagnostics.json")
        
        # Provide recommendations
        print("\nRecommendations:")
        if diagnostics["server_tests"]["status"] in ("already_running", "started"):
            if "endpoint_tests" in diagnostics and all(t["success"] for t in diagnostics["endpoint_tests"].values()):
                print("✅ MCP server is running correctly. All tests passed.")
            else:
                print("⚠️ MCP server is running but some endpoint tests failed.")
                print("   Check the logs and run with --autofix to apply fixes.")
        else:
            print("❌ MCP server is not running correctly.")
            print("   Run with --autofix to attempt automatic fixes.")
            print("   Check server_diagnostic.log for error details.")
            
    finally:
        # Make sure to stop our server if we started it
        if diagnostics["server_tests"].get("status") == "started":
            stop_server()

if __name__ == "__main__":
    main()
