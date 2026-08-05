#!/usr/bin/env python3
"""
Integration test script for IPFS Accelerate CLI and MCP server

This script tests the functionality of both the CLI tool and MCP server
to ensure they work consistently and share code properly.
"""

import os
import sys
import subprocess
import json
import time
import tempfile
from pathlib import Path


def test_cli_commands():
    """Test CLI commands"""
    print("🧪 Testing CLI commands...")

    # Test help
    print("  📋 Testing --help")
    result = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✅ Help command works")
    else:
        print(f"  ❌ Help command failed: {result.stderr}")
        return False

    # Test models list
    print("  📋 Testing models list")
    result = subprocess.run(
        [sys.executable, "cli.py", "models", "list", "--output-json"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            print(f"  ✅ Models list works: {len(data.get('models', []))} models found")
        except json.JSONDecodeError:
            print("  ❌ Models list output is not valid JSON")
            return False
    else:
        print(f"  ❌ Models list failed: {result.stderr}")
        return False

    # Test network status
    print("  📋 Testing network status")
    result = subprocess.run(
        [sys.executable, "cli.py", "network", "status", "--output-json"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            print(f"  ✅ Network status works: {data.get('status', 'unknown')}")
        except json.JSONDecodeError:
            print("  ❌ Network status output is not valid JSON")
            return False
    else:
        print(f"  ❌ Network status failed: {result.stderr}")
        return False

    # Test inference generate
    print("  📋 Testing inference generate")
    result = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "inference",
            "generate",
            "--prompt",
            "Hello world",
            "--model",
            "gpt2",
            "--output-json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            print(f"  ✅ Inference generate works: {data.get('success', False)}")
        except json.JSONDecodeError:
            print("  ❌ Inference generate output is not valid JSON")
            return False
    else:
        print(f"  ❌ Inference generate failed: {result.stderr}")
        return False

    # Test file add with temporary file
    print("  📋 Testing file add")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, IPFS!")
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, "cli.py", "files", "add", temp_file, "--output-json"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                print(f"  ✅ File add works: {data.get('success', False)}")
            except json.JSONDecodeError:
                print("  ❌ File add output is not valid JSON")
                return False
        else:
            print(f"  ❌ File add failed: {result.stderr}")
            return False
    finally:
        os.unlink(temp_file)

    return True


def test_shared_operations():
    """Test shared operations directly"""
    print("🧪 Testing shared operations...")

    try:
        from shared import (
            SharedCore,
            InferenceOperations,
            FileOperations,
            ModelOperations,
            NetworkOperations,
        )

        # Test shared core
        print("  📋 Testing SharedCore")
        core = SharedCore()
        status = core.get_status()
        print(f"  ✅ SharedCore works: uptime={status.get('uptime', 0):.1f}s")

        # Test inference operations
        print("  📋 Testing InferenceOperations")
        inference_ops = InferenceOperations(core)
        result = inference_ops.run_text_generation("gpt2", "Hello", max_length=50)
        print(f"  ✅ InferenceOperations works: {result.get('success', False)}")

        # Test model operations
        print("  📋 Testing ModelOperations")
        model_ops = ModelOperations(core)
        models = model_ops.list_models()
        print(f"  ✅ ModelOperations works: {len(models.get('models', []))} models")

        # Test network operations
        print("  📋 Testing NetworkOperations")
        network_ops = NetworkOperations(core)
        network_status = network_ops.get_network_status()
        print(f"  ✅ NetworkOperations works: {network_status.get('status', 'unknown')}")

        return True

    except ImportError as e:
        print(f"  ❌ Shared operations import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Shared operations test failed: {e}")
        return False


def test_mcp_server_startup():
    """Test MCP server startup (basic check)"""
    print("🧪 Testing MCP server startup...")

    # Start MCP server in background with timeout
    try:
        process = subprocess.Popen(
            [sys.executable, "cli.py", "mcp", "start", "--port", "8888"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait a bit for startup
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            print("  ✅ MCP server started successfully")
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"  ❌ MCP server failed to start: {stderr}")
            return False

    except Exception as e:
        print(f"  ❌ MCP server startup test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 Starting IPFS Accelerate Integration Tests\n")

    # Change to the correct directory
    os.chdir(Path(__file__).parent)

    tests_passed = 0
    total_tests = 3

    # Test CLI commands
    if test_cli_commands():
        tests_passed += 1
        print("✅ CLI commands test passed\n")
    else:
        print("❌ CLI commands test failed\n")

    # Test shared operations
    if test_shared_operations():
        tests_passed += 1
        print("✅ Shared operations test passed\n")
    else:
        print("❌ Shared operations test failed\n")

    # Test MCP server startup
    if test_mcp_server_startup():
        tests_passed += 1
        print("✅ MCP server startup test passed\n")
    else:
        print("❌ MCP server startup test failed\n")

    # Summary
    print("📊 Test Summary:")
    print(f"   Passed: {tests_passed}/{total_tests}")
    print(f"   Success rate: {tests_passed / total_tests * 100:.1f}%")

    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
