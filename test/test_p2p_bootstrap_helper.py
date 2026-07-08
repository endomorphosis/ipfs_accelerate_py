#!/usr/bin/env python3
"""
Test P2P Bootstrap Helper

Verifies that the simplified bootstrap helper works correctly
for peer discovery in GitHub Actions.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from ipfs_accelerate_py.github_cli.p2p_bootstrap_helper import SimplePeerBootstrap

def test_basic_initialization():
    """Test that bootstrap helper can be initialized"""
    print("Test 1: Basic initialization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
        
        assert helper.runner_name is not None, "Runner name should be detected"
        assert helper.cache_dir == Path(tmpdir), "Cache dir should match"
        
    print("  ✓ Bootstrap helper initialized successfully")
    return True

def test_peer_registration():
    """Test registering a peer"""
    print("Test 2: Peer registration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
        
        # Register a test peer
        peer_id = "QmTest123456789"
        listen_port = 9000
        public_ip = "192.168.1.100"
        multiaddr = f"/ip4/{public_ip}/tcp/{listen_port}/p2p/{peer_id}"
        
        result = helper.register_peer(peer_id, listen_port, multiaddr)
        assert result, "Peer registration should succeed"
        
        # Check that peer file was created
        peer_file = Path(tmpdir) / f"peer_{helper.runner_name}.json"
        assert peer_file.exists(), "Peer file should be created"
        
        # Verify peer file contents
        with open(peer_file, "r") as f:
            peer_info = json.load(f)
        
        assert peer_info["peer_id"] == peer_id, "Peer ID should match"
        assert peer_info["listen_port"] == listen_port, "Port should match"
        assert peer_info["multiaddr"] == multiaddr, "Multiaddr should match"
        
    print("  ✓ Peer registration works correctly")
    return True

def test_peer_discovery():
    """Test discovering registered peers"""
    print("Test 3: Peer discovery...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create helper for first peer
        helper1 = SimplePeerBootstrap(cache_dir=Path(tmpdir))
        original_name = helper1.runner_name
        
        # Register first peer
        peer1_id = "QmPeer1"
        helper1.register_peer(peer1_id, 9000, f"/ip4/192.168.1.1/tcp/9000/p2p/{peer1_id}")
        
        # Create helper for second peer (different runner name)
        helper2 = SimplePeerBootstrap(cache_dir=Path(tmpdir))
        helper2.runner_name = "different_runner"
        
        # Register second peer
        peer2_id = "QmPeer2"
        helper2.register_peer(peer2_id, 9001, f"/ip4/192.168.1.2/tcp/9001/p2p/{peer2_id}")
        
        # Discover peers from first helper
        discovered = helper1.discover_peers(max_peers=10)
        
        # Should find second peer but not itself
        assert len(discovered) == 1, f"Should discover 1 peer, found {len(discovered)}"
        assert discovered[0]["peer_id"] == peer2_id, "Should discover second peer"
        
    print("  ✓ Peer discovery works correctly")
    return True

def test_bootstrap_addrs():
    """Test getting bootstrap addresses"""
    print("Test 4: Bootstrap address retrieval...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and register multiple peers
        peers = []
        for i in range(3):
            helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
            helper.runner_name = f"runner_{i}"
            
            peer_id = f"QmPeer{i}"
            multiaddr = f"/ip4/192.168.1.{i+1}/tcp/900{i}/p2p/{peer_id}"
            helper.register_peer(peer_id, 9000 + i, multiaddr)
            peers.append(multiaddr)
        
        # Get bootstrap addresses from a new helper
        helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
        helper.runner_name = "new_runner"
        
        bootstrap_addrs = helper.get_bootstrap_addrs(max_peers=5)
        
        assert len(bootstrap_addrs) > 0, "Should find bootstrap peers"
        assert len(bootstrap_addrs) <= 3, "Should not exceed available peers"
        
    print("  ✓ Bootstrap address retrieval works correctly")
    return True

def test_environment_variable_bootstrap():
    """Test that environment variable bootstrap peers are included"""
    print("Test 5: Environment variable bootstrap...")
    
    # Set environment variable
    test_peer = "/ip4/10.0.0.1/tcp/9000/p2p/QmEnvPeer"
    os.environ["CACHE_BOOTSTRAP_PEERS"] = test_peer
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
            
            bootstrap_addrs = helper.get_bootstrap_addrs(max_peers=5)
            
            assert test_peer in bootstrap_addrs, "Environment variable peer should be included"
            
    finally:
        # Clean up
        if "CACHE_BOOTSTRAP_PEERS" in os.environ:
            del os.environ["CACHE_BOOTSTRAP_PEERS"]
    
    print("  ✓ Environment variable bootstrap works correctly")
    return True

def test_stale_peer_cleanup():
    """Test cleanup of stale peers"""
    print("Test 6: Stale peer cleanup...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        helper = SimplePeerBootstrap(cache_dir=Path(tmpdir), peer_ttl_minutes=0)
        
        # Register a peer
        peer_id = "QmStalePeer"
        helper.register_peer(peer_id, 9000, f"/ip4/192.168.1.1/tcp/9000/p2p/{peer_id}")
        
        # Cleanup should remove the stale peer (TTL is 0)
        import time
        time.sleep(0.1)  # Give it a moment to become stale
        
        cleaned = helper.cleanup_stale_peers()
        assert cleaned == 1, "Should clean up 1 stale peer"
        
        # Verify peer was removed
        peers = helper.discover_peers()
        assert len(peers) == 0, "Stale peer should be removed"
        
    print("  ✓ Stale peer cleanup works correctly")
    return True

def test_libp2p_bootstrap_fallback():
    """Test that standard libp2p bootstrap nodes are used when no peers are found"""
    print("Test 7: libp2p bootstrap fallback...")
    
    # Make sure no environment variable is set
    env_backup = os.environ.get("CACHE_BOOTSTRAP_PEERS")
    if "CACHE_BOOTSTRAP_PEERS" in os.environ:
        del os.environ["CACHE_BOOTSTRAP_PEERS"]
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = SimplePeerBootstrap(cache_dir=Path(tmpdir))
            
            # Get bootstrap addresses with no registered peers and no env var
            bootstrap_addrs = helper.get_bootstrap_addrs(max_peers=5)
            
            # Should fall back to libp2p bootstrap nodes
            assert len(bootstrap_addrs) > 0, "Should have bootstrap addresses"
            
            # Check that we got the libp2p bootstrap nodes
            expected_bootstrap_nodes = [
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
            ]
            
            # At least one of the expected bootstrap nodes should be in the list
            found_bootstrap_node = any(node in bootstrap_addrs for node in expected_bootstrap_nodes)
            assert found_bootstrap_node, "Should include at least one libp2p bootstrap node"
            
            # Should not exceed max_peers
            assert len(bootstrap_addrs) <= 5, f"Should not exceed max_peers (5), got {len(bootstrap_addrs)}"
            
    finally:
        # Restore environment variable if it was set
        if env_backup is not None:
            os.environ["CACHE_BOOTSTRAP_PEERS"] = env_backup
    
    print("  ✓ libp2p bootstrap fallback works correctly")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  P2P Bootstrap Helper Tests")
    print("="*60 + "\n")
    
    tests = [
        test_basic_initialization,
        test_peer_registration,
        test_peer_discovery,
        test_bootstrap_addrs,
        test_environment_variable_bootstrap,
        test_stale_peer_cleanup,
        test_libp2p_bootstrap_fallback
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ Test failed: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed with exception: {test.__name__}")
            print(f"     {e}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("✅ All tests passed!")
        print("="*60 + "\n")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
