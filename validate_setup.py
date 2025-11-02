#!/usr/bin/env python3
"""
Comprehensive Package Setup Validation
Tests all critical functionality of ipfs_accelerate_py
"""

import subprocess
import sys
import os
import time
import socket
import requests
from pathlib import Path


def run_command(cmd, timeout=30):
    """Run a command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_package_import():
    """Test basic package import"""
    print("🔍 Testing package import...")
    success, stdout, stderr = run_command("python -c 'import ipfs_accelerate_py; print(\"Import successful\")'")
    if success:
        print("✅ Package import: PASSED")
        return True
    else:
        print(f"❌ Package import: FAILED - {stderr}")
        return False


def test_cli_entry_points():
    """Test CLI entry points"""
    print("\n🔍 Testing CLI entry points...")
    
    # Test main CLI
    success, stdout, stderr = run_command("ipfs-accelerate --help")
    if success:
        print("✅ Main CLI entry point: PASSED")
    else:
        print(f"❌ Main CLI entry point: FAILED - {stderr}")
        return False
    
    # Test MCP commands
    success, stdout, stderr = run_command("ipfs-accelerate mcp start --help")
    if success:
        print("✅ MCP CLI commands: PASSED")
        return True
    else:
        print(f"❌ MCP CLI commands: FAILED - {stderr}")
        return False


def test_mcp_server():
    """Test MCP server functionality"""
    print("\n🔍 Testing MCP server...")
    
    # Find available port
    port = 9010
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex(('127.0.0.1', port)) == 0:
            port += 1
    
    print(f"   Starting MCP server on port {port}...")
    
    # Start MCP server in background
    process = subprocess.Popen([
        "ipfs-accelerate", "mcp", "start", 
        "--dashboard", "--host", "127.0.0.1", 
        "--port", str(port), "--keep-running"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test server response
        response = requests.get(f"http://127.0.0.1:{port}/", timeout=10)
        if response.status_code == 200:
            print("✅ MCP server: PASSED")
            success = True
        else:
            print(f"❌ MCP server: FAILED - HTTP {response.status_code}")
            success = False
    except requests.RequestException as e:
        print(f"❌ MCP server: FAILED - {e}")
        success = False
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    return success


def test_docker_functionality():
    """Test Docker functionality"""
    print("\n🔍 Testing Docker functionality...")
    
    # Test Docker access
    success, stdout, stderr = run_command("docker ps", timeout=10)
    if not success:
        print(f"❌ Docker access: FAILED - {stderr}")
        return False
    
    print("✅ Docker access: PASSED")
    
    # Test Docker build (quick test)
    print("   Testing Docker build...")
    build_cmd = "cd /home/barberb/ipfs_accelerate_py && docker build --platform linux/arm64 --target minimal -t ipfs-accelerate-py:setup-test . >/dev/null 2>&1"
    success, stdout, stderr = run_command(build_cmd, timeout=120)
    
    if success:
        print("✅ Docker build: PASSED")
        
        # Test container run
        print("   Testing container execution...")
        run_cmd = "docker run --platform linux/arm64 --rm ipfs-accelerate-py:setup-test ipfs-accelerate --help >/dev/null 2>&1"
        success, stdout, stderr = run_command(run_cmd, timeout=30)
        
        if success:
            print("✅ Docker container execution: PASSED")
            
            # Clean up test image
            run_command("docker rmi ipfs-accelerate-py:setup-test >/dev/null 2>&1")
            return True
        else:
            print(f"❌ Docker container execution: FAILED - {stderr}")
            return False
    else:
        print(f"❌ Docker build: FAILED - {stderr}")
        return False


def test_github_actions_readiness():
    """Test GitHub Actions CI/CD readiness"""
    print("\n🔍 Testing GitHub Actions readiness...")
    
    # Test sudo access
    success, stdout, stderr = run_command("sudo -n whoami")
    if success and "root" in stdout:
        print("✅ Passwordless sudo: PASSED")
    else:
        print("❌ Passwordless sudo: FAILED")
        return False
    
    # Test GitHub Actions runner service
    success, stdout, stderr = run_command("sudo systemctl is-active actions.runner.endomorphosis-ipfs_accelerate_py.arm64-dgx-spark-gb10-ipfs.service")
    if success and "active" in stdout:
        print("✅ GitHub Actions runner service: PASSED")
    else:
        print("❌ GitHub Actions runner service: FAILED")
        return False
    
    # Test Docker group membership
    success, stdout, stderr = run_command("groups $USER")
    if success and "docker" in stdout:
        print("✅ Docker group membership: PASSED")
        return True
    else:
        print("❌ Docker group membership: FAILED")
        return False


def main():
    """Main validation function"""
    print("🚀 IPFS Accelerate Python Package - Setup Validation")
    print("=" * 60)
    
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Architecture: {os.uname().machine}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    print()
    
    tests = [
        ("Package Import", test_package_import),
        ("CLI Entry Points", test_cli_entry_points), 
        ("MCP Server", test_mcp_server),
        ("Docker Functionality", test_docker_functionality),
        ("GitHub Actions Readiness", test_github_actions_readiness),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Package setup is complete and functional!")
        print("\n✅ Ready for:")
        print("   • Local development and testing")
        print("   • MCP server deployment")
        print("   • Docker containerization")
        print("   • GitHub Actions CI/CD")
        return 0
    else:
        print(f"⚠️  {failed} TESTS FAILED - Please address issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())