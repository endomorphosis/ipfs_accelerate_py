#!/usr/bin/env python3
"""
Integration tests for GitHub CLI and Copilot CLI integration

These tests verify that:
1. Python package imports work correctly
2. CLI commands are accessible
3. MCP tools are registered
"""

import sys
import os
import subprocess

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_status(message, success=True):
    """Print status message with color"""
    color = GREEN if success else RED
    symbol = "✓" if success else "✗"
    print(f"{color}{symbol}{RESET} {message}")

def print_section(title):
    """Print section header"""
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}{title}{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}\n")

def test_python_imports():
    """Test that Python package imports work"""
    print_section("Testing Python Package Imports")
    
    tests = [
        ("GitHub CLI wrapper", "from ipfs_accelerate_py.github_cli import GitHubCLI"),
        ("Workflow Queue", "from ipfs_accelerate_py.github_cli import WorkflowQueue"),
        ("Runner Manager", "from ipfs_accelerate_py.github_cli import RunnerManager"),
        ("Copilot CLI wrapper", "from ipfs_accelerate_py.copilot_cli import CopilotCLI"),
    ]
    
    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print_status(f"{name} import works", True)
            passed += 1
        except ImportError as e:
            print_status(f"{name} import failed: {e}", False)
    
    return passed, len(tests)

def test_cli_commands():
    """Test that CLI commands are accessible"""
    print_section("Testing CLI Commands")
    
    commands = [
        ("CLI help", ["python", "cli.py", "--help"]),
        ("GitHub subcommand", ["python", "cli.py", "github", "--help"]),
        ("GitHub auth", ["python", "cli.py", "github", "auth", "--help"]),
        ("GitHub repos", ["python", "cli.py", "github", "repos", "--help"]),
        ("GitHub workflows", ["python", "cli.py", "github", "workflows", "--help"]),
        ("GitHub queues", ["python", "cli.py", "github", "queues", "--help"]),
        ("GitHub runners", ["python", "cli.py", "github", "runners", "--help"]),
        ("Copilot subcommand", ["python", "cli.py", "copilot", "--help"]),
        ("Copilot suggest", ["python", "cli.py", "copilot", "suggest", "--help"]),
        ("Copilot explain", ["python", "cli.py", "copilot", "explain", "--help"]),
        ("Copilot git", ["python", "cli.py", "copilot", "git", "--help"]),
    ]
    
    passed = 0
    for name, cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if result.returncode == 0:
                print_status(f"{name} accessible", True)
                passed += 1
            else:
                print_status(f"{name} failed with code {result.returncode}", False)
        except Exception as e:
            print_status(f"{name} error: {e}", False)
    
    return passed, len(commands)

def test_mcp_tools():
    """Test that MCP tools are registered"""
    print_section("Testing MCP Tools Registration")
    
    try:
        # Try to import and check tool registration
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from ipfs_accelerate_py.mcp.tools import github_tools, copilot_tools
        
        # Check if registration functions exist
        tests = [
            ("GitHub tools module", hasattr(github_tools, 'register_github_tools')),
            ("Copilot tools module", hasattr(copilot_tools, 'register_copilot_tools')),
        ]
        
        passed = 0
        for name, result in tests:
            print_status(f"{name} found", result)
            if result:
                passed += 1
        
        return passed, len(tests)
        
    except Exception as e:
        print_status(f"MCP tools check failed: {e}", False)
        return 0, 2

def test_class_functionality():
    """Test basic class functionality"""
    print_section("Testing Class Functionality")
    
    passed = 0
    total = 3
    
    try:
        from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager
        
        # Test GitHubCLI instantiation
        try:
            # Note: This will fail if gh is not authenticated, but we're just testing instantiation
            gh = GitHubCLI()
            print_status("GitHubCLI instantiation works", True)
            passed += 1
        except Exception as e:
            print_status(f"GitHubCLI instantiation: {e}", False)
        
        # Test WorkflowQueue instantiation
        try:
            queue = WorkflowQueue()
            print_status("WorkflowQueue instantiation works", True)
            passed += 1
        except Exception as e:
            print_status(f"WorkflowQueue instantiation: {e}", False)
        
        # Test RunnerManager instantiation
        try:
            runner_mgr = RunnerManager()
            cores = runner_mgr.get_system_cores()
            print_status(f"RunnerManager works (detected {cores} cores)", True)
            passed += 1
        except Exception as e:
            print_status(f"RunnerManager: {e}", False)
        
    except Exception as e:
        print_status(f"Class functionality test failed: {e}", False)
    
    return passed, total

def main():
    """Run all integration tests"""
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}GitHub CLI and Copilot CLI Integration Tests{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    
    results = []
    
    # Run all tests
    results.append(test_python_imports())
    results.append(test_cli_commands())
    results.append(test_mcp_tools())
    results.append(test_class_functionality())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    
    # Print summary
    print_section("Test Summary")
    print(f"Total tests passed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print(f"\n{GREEN}✓ All tests passed!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed ({total_tests - total_passed} failures){RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
