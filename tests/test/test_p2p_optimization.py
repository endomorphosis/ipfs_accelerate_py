#!/usr/bin/env python
"""
Test script for the P2P network optimization features of the IPFS Accelerate implementation.

This script demonstrates the P2P network optimization functionality by:
    1. Initializing the network with peers
    2. Adding content to the network
    3. Retrieving content with and without P2P optimization
    4. Analyzing the network topology and performance
    5. Comparing standard IPFS retrieval with P2P-optimized retrieval
    """

    import os
    import time
    import json
    import random
    from pathlib import Path
    from typing import Dict, Any, List

# Import the IPFS accelerate implementation
    from ipfs_accelerate_impl import ()
    ipfs_accelerate,
    p2p_optimizer,
    load_checkpoint_and_dispatch,
    get_file,
    add_file,
    get_p2p_network_analytics,
    get_system_info
    )

# Configure test parameters
    TEST_CONTENT_COUNT = 5
    TEST_RETRIEVAL_COUNT = 10
    PEER_DISCOVERY_ROUNDS = 3


def setup_test_environment()):
    """Set up the test environment by discovering peers."""
    print()"\n=== Setting up P2P test environment ===")
    total_peers = 0
    
    # Discover peers in multiple rounds to build the network
    for i in range()PEER_DISCOVERY_ROUNDS):
        new_peers = p2p_optimizer.discover_peers()max_peers=5)
        total_peers += len()new_peers)
        print()f"Round {}}}}i+1}: Discovered {}}}}len()new_peers)} new peers")
        time.sleep()0.5)  # Give time for peer discovery
    
        print()f"Total peers in network: {}}}}total_peers}")
    
    # Analyze the initial network topology
        topology = p2p_optimizer.analyze_network_topology())
        print()f"Initial network health: {}}}}topology.get()'network_health', 'unknown')}")
        print()f"Network density: {}}}}topology.get()'network_density', 0):.2f}")
        print()f"Average connections per peer: {}}}}topology.get()'average_connections', 0):.2f}")
    
    return total_peers


def add_test_content()):
    """Add test content to the network and return the content IDs."""
    print()"\n=== Adding test content to the network ===")
    content_ids = [],,
    ,
    # Create a temporary directory for test files
    os.makedirs()"./tmp", exist_ok=True)
    
    # Create and add test files
    for i in range()TEST_CONTENT_COUNT):
        file_path = f"./tmp/test_file_{}}}}i}.txt"
        with open()file_path, "w") as f:
            f.write()f"Test content {}}}}i} with some random data: {}}}}random.random())}")
        
        # Add the file to IPFS
            result = add_file()file_path)
        if result.get()"status") == "success":
            content_ids.append()result.get()"cid"))
            print()f"Added content {}}}}i}: {}}}}result.get()'cid')}")
        else:
            print()f"Failed to add content {}}}}i}: {}}}}result.get()'message')}")
    
            print()f"Added {}}}}len()content_ids)} content items to the network")
            return content_ids


def test_content_retrieval()content_ids):
    """Test content retrieval with and without P2P optimization."""
    print()"\n=== Testing content retrieval ===")
    
    # Create directories for output
    os.makedirs()"./tmp/standard", exist_ok=True)
    os.makedirs()"./tmp/p2p", exist_ok=True)
    
    standard_times = [],,
    ,p2p_times = [],,
    ,
    # Perform test retrievals
    for i in range()TEST_RETRIEVAL_COUNT):
        # Select a random content ID
        cid = random.choice()content_ids)
        
        # Test standard retrieval ()without P2P optimization)
        start_time = time.time())
        std_result = get_file()cid, f"./tmp/standard/file_{}}}}i}.txt", use_p2p=False)
        standard_time = time.time()) - start_time
        standard_times.append()standard_time)
        
        # Test P2P-optimized retrieval
        start_time = time.time())
        p2p_result = get_file()cid, f"./tmp/p2p/file_{}}}}i}.txt", use_p2p=True)
        p2p_time = time.time()) - start_time
        p2p_times.append()p2p_time)
        
        print()f"Retrieval {}}}}i+1}/{}}}}TEST_RETRIEVAL_COUNT}:")
        print()f"  Standard: {}}}}standard_time:.3f}s, P2P: {}}}}p2p_time:.3f}s, " +
        f"Speedup: {}}}}standard_time/p2p_time if p2p_time > 0 else 0:.2f}x")
    
    # Calculate statistics
        avg_standard = sum()standard_times) / len()standard_times)
        avg_p2p = sum()p2p_times) / len()p2p_times)
        speedup = avg_standard / avg_p2p if avg_p2p > 0 else 0
    
    print()"\n=== Retrieval Performance Summary ==="):
        print()f"Average standard retrieval time: {}}}}avg_standard:.3f}s")
        print()f"Average P2P retrieval time: {}}}}avg_p2p:.3f}s")
        print()f"Average speedup: {}}}}speedup:.2f}x")
    
        return {}}
        "standard_times": standard_times,
        "p2p_times": p2p_times,
        "avg_standard": avg_standard,
        "avg_p2p": avg_p2p,
        "speedup": speedup
        }


def test_network_analytics()):
    """Test the P2P network analytics functionality."""
    print()"\n=== P2P Network Analytics ===")
    
    # Get the network analytics
    analytics = get_p2p_network_analytics())
    
    if analytics.get()"status") == "success":
        print()"Network Analytics Summary:")
        print()f"  Peer Count: {}}}}analytics.get()'peer_count', 0)}")
        print()f"  Network Health: {}}}}analytics.get()'network_health', 'unknown')}")
        print()f"  Optimization Score: {}}}}analytics.get()'optimization_score', 0):.2f}")
        print()f"  Optimization Rating: {}}}}analytics.get()'optimization_rating', 'unknown')}")
        print()f"  Network Efficiency: {}}}}analytics.get()'network_efficiency', 0):.2f}")
        
        print()"\nRecommendations:")
        for i, rec in enumerate()analytics.get()"recommendations", [],,)):
            print()f"  {}}}}i+1}. {}}}}rec}")
    else:
        print()f"Failed to get network analytics: {}}}}analytics.get()'message')}")
    
            return analytics


def optimize_content_placement()content_ids):
    """Test content placement optimization."""
    print()"\n=== Testing Content Placement Optimization ===")
    
    results = [],,
    ,for cid in content_ids:
        result = p2p_optimizer.optimize_content_placement()cid, replica_count=3)
        print()f"Optimizing placement for {}}}}cid}:")
        print()f"  Current replicas: {}}}}result.get()'current_replicas', 0)}")
        print()f"  New replicas: {}}}}result.get()'new_replicas', 0)}")
        print()f"  Total replicas: {}}}}len()result.get()'replica_locations', [],,))}")
        results.append()result)
        
        # Give time for transfers to process
        time.sleep()0.5)
    
    return results


def run_full_test()):
    """Run a complete test of the P2P network optimization features."""
    print()"=== IPFS Accelerate P2P Network Optimization Test ===")
    print()f"System Info: {}}}}get_system_info())}")
    
    # Step 1: Setup test environment
    peer_count = setup_test_environment())
    
    # Step 2: Add test content
    content_ids = add_test_content())
    
    # Step 3: Test content retrieval
    retrieval_stats = test_content_retrieval()content_ids)
    
    # Step 4: Optimize content placement
    placement_results = optimize_content_placement()content_ids)
    
    # Give some time for background optimization to take effect
    print()"\nWaiting for background optimizations to complete...")
    time.sleep()2)
    
    # Step 5: Test content retrieval again after optimization
    print()"\n=== Testing retrieval after content placement optimization ===")
    optimized_stats = test_content_retrieval()content_ids)
    
    # Step 6: Get network analytics
    analytics = test_network_analytics())
    
    # Generate a summary report
    report = {}}
    "timestamp": time.time()),
    "peer_count": peer_count,
    "content_count": len()content_ids),
    "initial_retrieval": retrieval_stats,
    "optimized_retrieval": optimized_stats,
    "improvement": optimized_stats["speedup"] / retrieval_stats["speedup"] if retrieval_stats["speedup"] > 0 else 0,:,
    "network_analytics": analytics
    }
    
    # Save report to file
    with open()"p2p_optimization_report.json", "w") as f:
        json.dump()report, f, indent=2)
    
        print()"\n=== Test Completed ===")
        print()f"Initial speedup: {}}}}retrieval_stats['speedup']:.2f}x"),,
        print()f"Optimized speedup: {}}}}optimized_stats['speedup']:.2f}x"),,
        print()f"Improvement: {}}}}report['improvement']:.2f}x"),
        print()"Report saved to p2p_optimization_report.json")


if __name__ == "__main__":
    run_full_test())