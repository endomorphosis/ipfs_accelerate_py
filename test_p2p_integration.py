#!/usr/bin/env python3
"""
Test P2P Integration - Verify Dashboard P2P Status Display

This script tests the complete integration from Python backend to dashboard frontend:
1. Tests backend API functions (get_peer_status, get_cache_stats)
2. Verifies frontend JavaScript functions exist
3. Checks API endpoints are accessible
4. Provides installation guidance if libp2p is not available
"""

import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_backend_functions():
    """Test the Python backend functions."""
    logger.info("\n=== Testing Backend Functions ===\n")
    
    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import get_peer_status, get_cache_stats
        
        # Test get_peer_status
        logger.info("1. Testing get_peer_status()...")
        peer_status = get_peer_status()
        logger.info(f"   Result: {json.dumps(peer_status, indent=6)}")
        
        if not peer_status.get('enabled'):
            logger.info("   ⚠️  P2P is disabled (libp2p not installed)")
            logger.info("   💡 To enable: pip install libp2p>=0.4.0 pymultihash>=0.8.2")
        else:
            logger.info("   ✅ P2P is enabled")
            logger.info(f"   📊 Connected peers: {peer_status.get('peer_count', 0)}")
        
        # Test get_cache_stats
        logger.info("\n2. Testing get_cache_stats()...")
        cache_stats = get_cache_stats()
        logger.info(f"   Result: {json.dumps({k: v for k, v in cache_stats.items() if k in ['available', 'p2p_enabled', 'p2p_peers', 'total_entries']}, indent=6)}")
        
        if cache_stats.get('available'):
            logger.info("   ✅ Cache is available")
        else:
            logger.info("   ❌ Cache is not available")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error testing backend functions: {e}")
        return False


def test_frontend_functions():
    """Test that frontend JavaScript functions are properly defined."""
    logger.info("\n=== Testing Frontend Functions ===\n")
    
    try:
        # Read dashboard.js
        with open('ipfs_accelerate_py/static/js/dashboard.js', 'r') as f:
            content = f.read()
        
        checks = [
            ('refreshPeerStatus() defined', 'async function refreshPeerStatus()' in content),
            ('refreshCacheStats() defined', 'async function refreshCacheStats()' in content),
            ('refreshPeerStatus() called in overview', 'refreshPeerStatus()' in content and 'case \'overview\':' in content),
            ('refreshCacheStats() called in overview', 'refreshCacheStats()' in content and 'case \'overview\':' in content),
            ('API endpoint /api/mcp/peers', '/api/mcp/peers' in content),
            ('API endpoint /api/mcp/cache/stats', '/api/mcp/cache/stats' in content),
            ('peer-status element updated', 'getElementById(\'peer-status\')' in content),
            ('peer-count element updated', 'getElementById(\'peer-count\')' in content),
            ('p2p-enabled element updated', 'getElementById(\'p2p-enabled\')' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                logger.info(f"   ✅ {check_name}")
            else:
                logger.info(f"   ❌ {check_name}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"   ❌ Error testing frontend functions: {e}")
        return False


def test_api_routes():
    """Test that Flask API routes exist."""
    logger.info("\n=== Testing API Routes ===\n")
    
    try:
        from ipfs_accelerate_py.mcp_dashboard import MCPDashboard
        import inspect
        
        # Create a minimal dashboard instance to check routes
        logger.info("1. Checking Flask routes exist...")
        
        # Read mcp_dashboard.py source
        with open('ipfs_accelerate_py/mcp_dashboard.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('/api/mcp/peers route', '@self.app.route(\'/api/mcp/peers\')' in content),
            ('/api/mcp/cache/stats route', '@self.app.route(\'/api/mcp/cache/stats\')' in content),
            ('get_peer_status import', 'from ipfs_accelerate_py.mcp.tools.dashboard_data import get_peer_status' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                logger.info(f"   ✅ {check_name}")
            else:
                logger.info(f"   ❌ {check_name}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"   ❌ Error testing API routes: {e}")
        return False


def check_libp2p_installation():
    """Check if libp2p and dependencies are installed."""
    logger.info("\n=== Checking libp2p Installation ===\n")
    
    dependencies = [
        ('libp2p', 'libp2p', 'pip install libp2p>=0.4.0'),
        ('pymultihash', 'pymultihash', 'pip install pymultihash>=0.8.2'),
        ('multiformats', 'multiformats', 'pip install multiformats>=0.3.0'),
    ]
    
    all_installed = True
    for name, module, install_cmd in dependencies:
        try:
            __import__(module)
            logger.info(f"   ✅ {name} is installed")
        except ImportError:
            logger.info(f"   ⚠️  {name} is NOT installed")
            logger.info(f"      Install with: {install_cmd}")
            all_installed = False
    
    return all_installed


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("P2P Integration Test Suite")
    logger.info("=" * 70)
    
    results = {
        'Backend Functions': test_backend_functions(),
        'Frontend Functions': test_frontend_functions(),
        'API Routes': test_api_routes(),
        'libp2p Installation': check_libp2p_installation(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("Test Results Summary")
    logger.info("=" * 70 + "\n")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "⚠️  NEEDS ATTENTION"
        logger.info(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ All tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. Start MCP server: python3 -m ipfs_accelerate_py.mcp_dashboard")
        logger.info("2. Open browser: http://localhost:8899")
        logger.info("3. Check 'Overview' tab -> '🌐 P2P Peer System' section")
    else:
        logger.info("⚠️  Some tests need attention")
        logger.info("\nTo fix:")
        logger.info("1. Install missing dependencies (see above)")
        logger.info("2. Re-run this test script")
        logger.info("3. If issues persist, check P2P_SETUP_GUIDE.md")
    logger.info("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
