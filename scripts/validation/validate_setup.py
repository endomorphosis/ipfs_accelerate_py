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
import shutil
import tempfile
from typing import Optional
import requests
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_cli_command():
    """Resolve the CLI entry command from the active environment or repo checkout."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = Path(virtual_env) / "bin" / "ipfs-accelerate"
        if candidate.exists():
            return [str(candidate)]

    local_candidate = REPO_ROOT / ".venv" / "bin" / "ipfs-accelerate"
    if local_candidate.exists():
        return [str(local_candidate)]

    installed = shutil.which("ipfs-accelerate")
    if installed:
        return [installed]

    return [sys.executable, "-m", "ipfs_accelerate_py.cli_entry"]


CLI_COMMAND = _resolve_cli_command()


def wait_for_http_ready(base_url: str, process: subprocess.Popen, timeout: float = 30.0) -> tuple[bool, Optional[str]]:
    """Poll the MCP HTTP surface until it is ready or the process exits."""
    deadline = time.time() + timeout
    urls = [f"{base_url}/", f"{base_url}/health", f"{base_url}/dashboard"]
    last_error: Optional[str] = None

    while time.time() < deadline:
        if process.poll() is not None:
            return False, f"server process exited early with code {process.returncode}"

        for url in urls:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True, None
                last_error = f"{url} returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)

        time.sleep(0.5)

    return False, last_error or f"server did not become ready within {timeout:.0f}s"


def run_command(cmd, timeout=30):
    """Run a command and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_package_import():
    """Test basic package import"""
    print("🔍 Testing package import...")
    success, stdout, stderr = run_command(
        [sys.executable, "-c", "import ipfs_accelerate_py; print('Import successful')"]
    )
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
    success, stdout, stderr = run_command(CLI_COMMAND + ["--help"])
    if success:
        print("✅ Main CLI entry point: PASSED")
    else:
        print(f"❌ Main CLI entry point: FAILED - {stderr}")
        return False
    
    # Test MCP commands
    success, stdout, stderr = run_command(CLI_COMMAND + ["mcp", "start", "--help"])
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
    
    stdout_log = tempfile.TemporaryFile(mode="w+t")
    stderr_log = tempfile.TemporaryFile(mode="w+t")

    # Start MCP server in background without blocking on verbose startup logs.
    process = subprocess.Popen(
        CLI_COMMAND + [
            "mcp", "start",
            "--dashboard", "--host", "127.0.0.1",
            "--port", str(port), "--keep-running",
        ],
        stdout=stdout_log,
        stderr=stderr_log,
        cwd=REPO_ROOT,
        text=True,
    )
    
    base_url = f"http://127.0.0.1:{port}"
    
    try:
        ready, error = wait_for_http_ready(base_url, process, timeout=30.0)
        if not ready:
            stdout_log.seek(0)
            stderr_log.seek(0)
            stdout_data = stdout_log.read()
            stderr_data = stderr_log.read()
            details = error or "unknown startup failure"
            if stderr_data.strip():
                details = f"{details} | stderr: {stderr_data.strip()}"
            elif stdout_data.strip():
                details = f"{details} | stdout: {stdout_data.strip()}"
            print(f"❌ MCP server: FAILED - {details}")
            return False

        response = requests.get(f"{base_url}/", timeout=10)
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
        stdout_log.close()
        stderr_log.close()
    
    return success


def test_docker_functionality():
    """Test Docker functionality"""
    print("\n🔍 Testing Docker functionality...")

    if not shutil.which("docker"):
        print("⏭️  Docker functionality: SKIPPED - docker is not installed")
        return None
    
    # Test Docker access
    success, stdout, stderr = run_command(["docker", "ps"], timeout=10)
    if not success:
        print(f"⏭️  Docker functionality: SKIPPED - {stderr.strip() or 'docker is not accessible'}")
        return None
    
    print("✅ Docker access: PASSED")
    
    # Test Docker build (quick test)
    print("   Testing Docker build...")
    success, stdout, stderr = run_command(
        [
            "docker", "build", "--platform", "linux/arm64", "--target", "minimal",
            "-t", "ipfs-accelerate-py:setup-test", ".",
        ],
        timeout=120,
    )
    
    if success:
        print("✅ Docker build: PASSED")
        
        # Test container run
        print("   Testing container execution...")
        success, stdout, stderr = run_command(
            [
                "docker", "run", "--platform", "linux/arm64", "--rm",
                "ipfs-accelerate-py:setup-test", "ipfs-accelerate", "--help",
            ],
            timeout=30,
        )
        
        if success:
            print("✅ Docker container execution: PASSED")
            
            # Clean up test image
            run_command(["docker", "rmi", "ipfs-accelerate-py:setup-test"])
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

    if not shutil.which("sudo") or not shutil.which("systemctl"):
        print("⏭️  GitHub Actions readiness: SKIPPED - sudo/systemctl unavailable")
        return None
    
    # Test sudo access
    success, stdout, stderr = run_command(["sudo", "-n", "whoami"])
    if success and "root" in stdout:
        print("✅ Passwordless sudo: PASSED")
    else:
        print("⏭️  GitHub Actions readiness: SKIPPED - passwordless sudo unavailable")
        return None
    
    # Test GitHub Actions runner service
    success, stdout, stderr = run_command(
        [
            "sudo", "systemctl", "is-active",
            "actions.runner.endomorphosis-ipfs_accelerate_py.arm64-dgx-spark-gb10-ipfs.service",
        ]
    )
    if success and "active" in stdout:
        print("✅ GitHub Actions runner service: PASSED")
    else:
        print("❌ GitHub Actions runner service: FAILED")
        return False
    
    # Test Docker group membership
    success, stdout, stderr = run_command(["groups", os.getenv("USER", "")])
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
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is True:
                passed += 1
            elif result is None:
                skipped += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Package setup is complete and functional!")
        if skipped:
            print(f"\nℹ️  {skipped} environment-specific checks were skipped")
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