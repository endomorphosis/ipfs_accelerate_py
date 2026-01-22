#!/usr/bin/env python3
"""
Universal Connectivity Test for IPFS Accelerate

This test demonstrates libp2p connectivity following the universal-connectivity
pattern from https://github.com/libp2p/universal-connectivity

Tests:
1. Create libp2p host and listen on a port
2. Display peer ID and multiaddr for sharing
3. Accept peer connections from other instances
4. Demonstrate P2P communication

Usage:
  python test_universal_connectivity.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('universal_connectivity_test')


async def create_host(port: int = 9200):
    """Create a libp2p host and listen on specified port."""
    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    
    logger.info("Ensuring libp2p compatibility...")
    if not ensure_libp2p_compatible():
        logger.error("Failed to ensure libp2p compatibility")
        return None
    
    from libp2p import new_host
    from multiaddr import Multiaddr
    
    # Create host listening on all interfaces
    listen_addr = Multiaddr(f'/ip4/0.0.0.0/tcp/{port}')
    host = new_host(listen_addrs=[listen_addr])
    
    peer_id = host.get_id().pretty()
    logger.info(f"✓ Created libp2p host")
    logger.info(f"  Peer ID: {peer_id}")
    logger.info(f"  Listening on: {listen_addr}")
    
    # Get multiaddr with peer ID for sharing
    multiaddr_str = f"/ip4/0.0.0.0/tcp/{port}/p2p/{peer_id}"
    logger.info(f"  Full multiaddr: {multiaddr_str}")
    
    return host


async def test_host_creation():
    """Test 1: Create libp2p host"""
    logger.info("="*70)
    logger.info("TEST 1: Creating libp2p host")
    logger.info("="*70)
    
    try:
        host = await create_host(port=9200)
        if host:
            logger.info("✓ TEST PASSED: Host created successfully")
            return True, host
        else:
            logger.error("✗ TEST FAILED: Could not create host")
            return False, None
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_peer_info(host):
    """Test 2: Display peer information for connectivity"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Peer Information")
    logger.info("="*70)
    
    try:
        peer_id = host.get_id().pretty()
        
        logger.info("\n📋 Connection Information:")
        logger.info("-" * 70)
        logger.info(f"Peer ID: {peer_id}")
        logger.info(f"\nTo connect from another instance, use:")
        logger.info(f"  /ip4/127.0.0.1/tcp/9200/p2p/{peer_id}")
        logger.info(f"\nOr from a remote machine (replace <your-ip>):")
        logger.info(f"  /ip4/<your-ip>/tcp/9200/p2p/{peer_id}")
        logger.info("-" * 70)
        
        logger.info("\n✓ TEST PASSED: Peer info displayed")
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


async def test_listen_for_connections(host, duration: int = 10):
    """Test 3: Listen for incoming connections"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Listening for connections")
    logger.info("="*70)
    
    try:
        logger.info(f"\nListening for {duration} seconds...")
        logger.info("Waiting for peer connections...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Get current network state
            network = host.get_network()
            connections = network.connections
            
            if connections:
                logger.info(f"\n✓ Connected to {len(connections)} peer(s):")
                for peer_id, conn in connections.items():
                    logger.info(f"  - {peer_id.pretty()}")
            
            await asyncio.sleep(2)
        
        logger.info("\n✓ TEST PASSED: Listening completed")
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


async def test_connect_to_peer(host, peer_multiaddr: str):
    """Test 4: Connect to a specific peer"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Connecting to peer")
    logger.info("="*70)
    
    try:
        from multiaddr import Multiaddr
        from libp2p.peer.peerinfo import info_from_p2p_addr
        
        logger.info(f"Connecting to: {peer_multiaddr}")
        
        # Convert to Multiaddr
        addr = Multiaddr(peer_multiaddr)
        peer_info = info_from_p2p_addr(addr)
        
        # Connect
        await host.connect(peer_info)
        
        logger.info(f"✓ Successfully connected to {peer_info.peer_id.pretty()}")
        logger.info("✓ TEST PASSED: Connection established")
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_interactive_mode():
    """Run in interactive mode for manual testing"""
    logger.info("\n" + "="*70)
    logger.info("INTERACTIVE MODE: Universal Connectivity Test")
    logger.info("="*70)
    
    # Create host
    success, host = await test_host_creation()
    if not success or not host:
        logger.error("Failed to create host, exiting")
        return False
    
    # Display peer info
    await test_peer_info(host)
    
    # Ask if user wants to connect to a peer
    logger.info("\n")
    try:
        user_input = input("Do you want to connect to a peer? (y/N): ").strip().lower()
        
        if user_input == 'y':
            peer_addr = input("Enter peer multiaddr: ").strip()
            if peer_addr:
                await test_connect_to_peer(host, peer_addr)
        
        # Listen for connections
        logger.info("\n")
        duration = 30  # Listen for 30 seconds
        await test_listen_for_connections(host, duration)
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    
    logger.info("\n" + "="*70)
    logger.info("Test completed")
    logger.info("="*70)
    
    return True


async def run_automated_tests():
    """Run automated tests"""
    logger.info("\n" + "="*70)
    logger.info("AUTOMATED TESTS: Universal Connectivity")
    logger.info("="*70)
    
    results = []
    
    # Test 1: Create host
    success, host = await test_host_creation()
    results.append(("Host Creation", success))
    
    if not success or not host:
        logger.error("Cannot continue without host")
        return False
    
    # Test 2: Peer info
    success = await test_peer_info(host)
    results.append(("Peer Information", success))
    
    # Test 3: Listen for connections (brief)
    success = await test_listen_for_connections(host, duration=5)
    results.append(("Listen for Connections", success))
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("="*70)
    
    return passed == total


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Connectivity Test for IPFS Accelerate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run automated tests
  python test_universal_connectivity.py --automated
  
  # Run in interactive mode (default)
  python test_universal_connectivity.py
  
  # Specify custom port
  python test_universal_connectivity.py --port 9300
        """
    )
    
    parser.add_argument(
        '--automated',
        action='store_true',
        help='Run automated tests instead of interactive mode'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=9200,
        help='Port to listen on (default: 9200)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    if args.automated:
        success = asyncio.run(run_automated_tests())
    else:
        success = asyncio.run(run_interactive_mode())
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
