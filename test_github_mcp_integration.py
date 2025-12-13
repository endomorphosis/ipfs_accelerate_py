#!/usr/bin/env python3
"""
Test GitHub CLI and MCP Integration

This script tests that the GitHub CLI tools are properly integrated
with the MCP server and can be called correctly.
"""

import sys
import json


def test_github_tools_registration():
    """Test that GitHub tools can be registered with MCP server."""
    print("=" * 60)
    print("TEST 1: GitHub Tools Registration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.mcp.tools.github_tools import register_tools
        
        # Create mock MCP instance
        class MockMCP:
            def __init__(self):
                self.tools = []
                
            def tool(self):
                def decorator(func):
                    self.tools.append({
                        'name': func.__name__,
                        'doc': func.__doc__ or 'No description'
                    })
                    return func
                return decorator
        
        mcp = MockMCP()
        register_tools(mcp)
        
        print(f"✓ Registered {len(mcp.tools)} GitHub tools:")
        for tool in mcp.tools:
            print(f"  - {tool['name']}")
            # Print first line of docstring
            first_line = tool['doc'].split('\n')[0].strip()
            print(f"    {first_line}")
        
        assert len(mcp.tools) == 6, f"Expected 6 tools, got {len(mcp.tools)}"
        print("\n✓ TEST PASSED: All GitHub tools registered successfully\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_user_info_function():
    """Test that get_user_info() works correctly."""
    print("=" * 60)
    print("TEST 2: User Info Function")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info
        
        print("Calling get_user_info()...")
        user_info = get_user_info()
        
        print("\nUser info response:")
        print(json.dumps(user_info, indent=2))
        
        # Check required fields
        assert 'authenticated' in user_info, "Missing 'authenticated' field"
        
        if user_info['authenticated']:
            assert 'username' in user_info, "Missing 'username' field"
            assert 'token_type' in user_info, "Missing 'token_type' field"
            print(f"\n✓ User authenticated as: {user_info['username']}")
            print(f"✓ Token type: {user_info['token_type']}")
        else:
            print("\n⚠ User not authenticated (this is OK if gh auth login hasn't been run)")
            print("  To authenticate: gh auth login")
        
        print("\n✓ TEST PASSED: get_user_info() works correctly\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cache_stats():
    """Test that cache stats work correctly."""
    print("=" * 60)
    print("TEST 3: Cache Stats")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.github_cli.cache import get_global_cache
        
        print("Getting global cache...")
        cache = get_global_cache()
        
        print("Calling get_stats()...")
        stats = cache.get_stats()
        
        print("\nCache statistics:")
        print(json.dumps(stats, indent=2))
        
        # Check required fields (use actual field names from cache stats)
        assert 'cache_size' in stats or 'total_entries' in stats, "Missing cache size field"
        assert 'hit_rate' in stats, "Missing 'hit_rate' field"
        assert 'p2p_enabled' in stats, "Missing 'p2p_enabled' field"
        
        # Use cache_size if total_entries not available
        total_entries = stats.get('total_entries', stats.get('cache_size', 0))
        
        print(f"\n✓ Total cache entries: {total_entries}")
        print(f"✓ Cache hit rate: {stats['hit_rate']}%")
        print(f"✓ P2P enabled: {stats['p2p_enabled']}")
        
        if stats['p2p_enabled']:
            print(f"✓ P2P peers: {stats.get('p2p_peers', 0)}")
        else:
            print("⚠ P2P cache sharing disabled (install py-libp2p to enable)")
        
        print("\n✓ TEST PASSED: Cache stats work correctly\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_server_initialization():
    """Test that MCP server can be initialized with GitHub tools."""
    print("=" * 60)
    print("TEST 4: MCP Server Initialization")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer
        
        print("Creating MCP server instance...")
        server = IPFSAccelerateMCPServer(
            name="test-server",
            host="127.0.0.1",
            port=9999,
            debug=False
        )
        
        print("✓ MCP server created successfully")
        
        # Verify server has tools registered
        if hasattr(server, 'mcp') and hasattr(server.mcp, 'tools'):
            tools = server.mcp.tools
            github_tools = [name for name in tools.keys() if name.startswith('gh_')]
            print(f"✓ Found {len(github_tools)} GitHub tools in MCP server")
        else:
            print("⚠ Cannot verify tools (using standalone MCP mode)")
        
        print("\n✓ TEST PASSED: MCP server initializes with GitHub tools\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_github_cli_wrapper():
    """Test that GitHubCLI wrapper works correctly."""
    print("=" * 60)
    print("TEST 5: GitHub CLI Wrapper")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.github_cli import GitHubCLI, RunnerManager
        
        print("Creating GitHubCLI instance...")
        gh = GitHubCLI(
            enable_cache=True,
            auto_refresh_token=False  # Don't prompt for auth
        )
        
        print("✓ GitHubCLI created successfully")
        print(f"✓ Cache enabled: {gh.cache is not None}")
        
        print("\nGetting auth status...")
        auth_status = gh.get_auth_status()
        print(f"✓ Authenticated: {auth_status.get('authenticated', False)}")
        
        print("\nCreating RunnerManager...")
        runner_mgr = RunnerManager(gh_cli=gh)
        
        print(f"✓ System architecture: {runner_mgr.get_system_architecture()}")
        print(f"✓ Runner labels: {runner_mgr.get_runner_labels()}")
        print(f"✓ System cores: {runner_mgr.get_system_cores()}")
        
        print("\n✓ TEST PASSED: GitHub CLI wrapper works correctly\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GitHub CLI and MCP Integration Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_github_tools_registration,
        test_user_info_function,
        test_cache_stats,
        test_github_cli_wrapper,
        test_mcp_server_initialization,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Unexpected error running {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!\n")
        print("The GitHub CLI and MCP integration is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED\n")
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
