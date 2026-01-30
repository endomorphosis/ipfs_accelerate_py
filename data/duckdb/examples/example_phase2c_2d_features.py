"""
Comprehensive Example: Phase 2C & 2D Features

This example demonstrates all Phase 2C (Distributed Features) and Phase 2D
(Advanced Features) capabilities including:
- Distributed query execution
- P2P database synchronization
- Knowledge graph population
- Query optimization
- Performance monitoring
"""

import time
from typing import Dict, Any, List

# Import Phase 2C & 2D modules
try:
    from data.duckdb.ipfs_integration import (
        # Phase 2A & 2B
        IPFSConfig,
        IPFSStorage,
        IPFSDBBackend,
        BenchmarkKnowledgeGraph,
        IPFSCacheManager,
        # Phase 2C - Distributed Features
        DistributedQueryExecutor,
        P2PSynchronizer,
        KnowledgeGraphPopulator,
        # Phase 2D - Advanced Features
        QueryOptimizer,
        PerformanceMonitor
    )
    from data.duckdb.core.benchmark_db_api_ipfs import BenchmarkDBAPIIPFS
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the repository root")
    exit(1)


def example_distributed_query_execution():
    """
    Example 1: Distributed Query Execution
    
    Demonstrates executing queries across multiple IPFS nodes with
    automatic result aggregation.
    """
    print("\n" + "="*60)
    print("Example 1: Distributed Query Execution")
    print("="*60)
    
    # Initialize distributed query executor
    executor = DistributedQueryExecutor(max_workers=4)
    
    # Register IPFS nodes for distributed execution
    executor.register_node('node1', {
        'endpoint': 'http://node1.example.com:5001',
        'capabilities': ['gpu', 'cpu'],
        'region': 'us-west'
    })
    executor.register_node('node2', {
        'endpoint': 'http://node2.example.com:5001',
        'capabilities': ['cpu'],
        'region': 'us-east'
    })
    executor.register_node('node3', {
        'endpoint': 'http://node3.example.com:5001',
        'capabilities': ['gpu'],
        'region': 'eu-west'
    })
    
    # Execute distributed query
    query = "SELECT model_name, AVG(throughput) FROM benchmarks GROUP BY model_name"
    result = executor.execute_distributed_query(
        query,
        partition_strategy='round_robin'
    )
    
    print(f"Query executed across {result['node_count']} nodes")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"Distributed: {result['distributed']}")
    
    # Execute map-reduce operation
    def map_func(data):
        return {'max_throughput': max([x.get('throughput', 0) for x in data] or [0])}
    
    def reduce_func(results):
        return {'global_max': max([r.get('max_throughput', 0) for r in results])}
    
    mr_result = executor.execute_map_reduce(
        query,
        map_func=map_func,
        reduce_func=reduce_func
    )
    
    print(f"\nMap-Reduce Result: {mr_result['results']}")
    
    # Get cluster status
    status = executor.get_cluster_status()
    print(f"\nCluster Status:")
    print(f"  Total nodes: {status['total_nodes']}")
    print(f"  Available nodes: {status['available_nodes']}")
    print(f"  Average load: {status['average_load']:.2f}")


def example_p2p_synchronization():
    """
    Example 2: P2P Database Synchronization
    
    Demonstrates peer-to-peer synchronization of databases across
    IPFS nodes with conflict resolution.
    """
    print("\n" + "="*60)
    print("Example 2: P2P Database Synchronization")
    print("="*60)
    
    # Initialize P2P synchronizer
    synchronizer = P2PSynchronizer()
    
    # Register peers
    synchronizer.register_peer('peer1', {
        'endpoint': 'http://peer1.example.com:5001',
        'databases': ['benchmarks.db', 'models.db']
    })
    synchronizer.register_peer('peer2', {
        'endpoint': 'http://peer2.example.com:5001',
        'databases': ['benchmarks.db']
    })
    
    # Sync with specific peer
    db_cid = 'QmExampleDatabaseCID123'
    result = synchronizer.sync_with_peer('peer1', db_cid, force=True)
    
    print(f"Sync Status: {result['status']}")
    print(f"Peer: {result.get('peer_id', 'N/A')}")
    
    # Sync with all peers
    all_results = synchronizer.sync_all_peers(db_cid)
    print(f"\nAll Peers Sync:")
    print(f"  Total peers: {all_results['total_peers']}")
    print(f"  Synced: {all_results['synced']}")
    print(f"  Failed: {all_results['failed']}")
    print(f"  Skipped: {all_results['skipped']}")
    
    # Conflict resolution
    local_cid = 'QmLocalVersionCID'
    remote_cid = 'QmRemoteVersionCID'
    
    try:
        resolved_cid = synchronizer.resolve_conflicts(
            local_cid,
            remote_cid,
            strategy='latest'
        )
        print(f"\nResolved CID: {resolved_cid}")
    except Exception as e:
        print(f"\nConflict resolution: {e}")
    
    # Get sync status
    sync_status = synchronizer.get_sync_status()
    print(f"\nSync Status:")
    print(f"  Total peers: {sync_status['total_peers']}")
    print(f"  Active peers: {sync_status['active_peers']}")


def example_knowledge_graph_population():
    """
    Example 3: Knowledge Graph Population
    
    Demonstrates populating knowledge graph from benchmark data,
    discovering relationships, and performing semantic search.
    """
    print("\n" + "="*60)
    print("Example 3: Knowledge Graph Population")
    print("="*60)
    
    # Initialize knowledge graph and populator
    kg = BenchmarkKnowledgeGraph()
    populator = KnowledgeGraphPopulator(knowledge_graph=kg)
    
    # Sample benchmark data
    benchmarks = [
        {
            'model_name': 'bert-base',
            'model_family': 'transformer',
            'hardware_type': 'cuda',
            'throughput': 1250.5,
            'latency': 25.6,
            'framework': 'pytorch',
            'timestamp': time.time()
        },
        {
            'model_name': 'bert-large',
            'model_family': 'transformer',
            'hardware_type': 'cuda',
            'throughput': 850.3,
            'latency': 38.2,
            'framework': 'pytorch',
            'timestamp': time.time()
        },
        {
            'model_name': 'gpt2-small',
            'model_family': 'gpt',
            'hardware_type': 'cuda',
            'throughput': 980.1,
            'latency': 32.4,
            'framework': 'pytorch',
            'timestamp': time.time()
        },
        {
            'model_name': 'resnet50',
            'model_family': 'cnn',
            'hardware_type': 'cuda',
            'throughput': 2100.0,
            'latency': 15.2,
            'framework': 'tensorflow',
            'timestamp': time.time()
        }
    ]
    
    # Populate knowledge graph
    stats = populator.populate_from_benchmarks(
        benchmarks,
        discover_relationships=True
    )
    
    print(f"Population Stats:")
    print(f"  Nodes added: {stats['nodes_added']}")
    print(f"  Edges added: {stats['edges_added']}")
    print(f"  Relationships discovered: {stats['relationships_discovered']}")
    print(f"  Execution time: {stats['execution_time']:.2f}s")
    
    # Find similar benchmarks
    benchmark_id = 'bert-base_cuda_' + str(int(benchmarks[0]['timestamp']))
    similar = populator.find_similar_benchmarks(benchmark_id, top_k=3)
    
    print(f"\nSimilar to {benchmark_id}:")
    for sim in similar:
        print(f"  {sim['node_id']}: {sim['similarity']:.2f}")
    
    # Detect performance patterns
    patterns = populator.detect_performance_patterns(benchmarks)
    print(f"\nPerformance Patterns Detected: {len(patterns)}")
    for pattern in patterns:
        print(f"  {pattern['hardware_type']}: avg {pattern.get('avg_throughput', 0):.1f} ops/s")
    
    # Create hardware compatibility map
    compat_map = populator.create_hardware_compatibility_map(benchmarks)
    print(f"\nHardware Compatibility:")
    for model, hardware_list in compat_map.items():
        print(f"  {model}: {', '.join(hardware_list)}")
    
    # Get graph statistics
    graph_stats = populator.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {graph_stats['total_nodes']}")
    print(f"  Total edges: {graph_stats['total_edges']}")


def example_query_optimization():
    """
    Example 4: Query Optimization
    
    Demonstrates query optimization with intelligent caching
    and performance analysis.
    """
    print("\n" + "="*60)
    print("Example 4: Query Optimization")
    print("="*60)
    
    # Initialize optimizer with cache manager
    cache_manager = IPFSCacheManager()
    optimizer = QueryOptimizer(cache_manager=cache_manager)
    
    # Optimize query
    query = """
        SELECT model_name, hardware_type, AVG(throughput) as avg_throughput
        FROM benchmarks
        WHERE hardware_type = 'cuda' AND throughput > 1000
        ORDER BY avg_throughput DESC
        LIMIT 10
    """
    
    result = optimizer.optimize_query(query)
    
    print(f"Query Optimization:")
    print(f"  Optimizations applied: {', '.join(result['optimizations_applied']) or 'None'}")
    print(f"  Cached: {result['cached']}")
    print(f"  Optimization time: {result['optimization_time']*1000:.2f}ms")
    
    # Analyze query performance
    execution_time = 0.125  # Simulated
    result_count = 10
    
    analysis = optimizer.analyze_query_performance(
        query,
        execution_time,
        result_count
    )
    
    print(f"\nPerformance Analysis:")
    print(f"  Execution time: {analysis['execution_time']:.3f}s")
    print(f"  Average time: {analysis['average_time']:.3f}s")
    print(f"  Performance ratio: {analysis['performance_ratio']:.2f}x")
    print(f"  Suggestions: {len(analysis['suggestions'])}")
    
    for suggestion in analysis['suggestions']:
        print(f"    [{suggestion['severity']}] {suggestion['message']}")
    
    # Get caching strategy
    strategy = optimizer.get_caching_strategy(
        query,
        execution_time=0.125,
        result_size=1024 * 50  # 50KB
    )
    
    print(f"\nCaching Strategy:")
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  TTL: {strategy['ttl']}s")
    print(f"  Prefetch: {strategy['prefetch']}")
    print(f"  Reasons: {strategy['reasons']}")
    
    # Get optimizer statistics
    opt_stats = optimizer.get_optimizer_statistics()
    print(f"\nOptimizer Statistics:")
    print(f"  Total queries optimized: {opt_stats['total_queries_optimized']}")
    print(f"  Total executions: {opt_stats['total_query_executions']}")


def example_performance_monitoring():
    """
    Example 5: Performance Monitoring
    
    Demonstrates real-time performance monitoring, trend analysis,
    and alerting.
    """
    print("\n" + "="*60)
    print("Example 5: Performance Monitoring")
    print("="*60)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Record some metrics
    for i in range(10):
        monitor.record_metric('query_time', 0.5 + i * 0.1, {'query_id': f'q{i}'})
        monitor.record_metric('cache_hit_rate', 0.8 - i * 0.05, {'query_id': f'q{i}'})
        time.sleep(0.1)
    
    # Get metric summary
    query_summary = monitor.get_metric_summary('query_time', time_window=3600)
    print(f"Query Time Summary:")
    print(f"  Count: {query_summary.get('count', 0)}")
    print(f"  Average: {query_summary.get('average', 0):.3f}s")
    print(f"  Min: {query_summary.get('min', 0):.3f}s")
    print(f"  Max: {query_summary.get('max', 0):.3f}s")
    
    # Analyze trends
    trends = monitor.get_performance_trends(['query_time', 'cache_hit_rate'])
    print(f"\nPerformance Trends:")
    for metric_name, trend_data in trends.items():
        print(f"  {metric_name}:")
        print(f"    Trend: {trend_data['trend']}")
        print(f"    Change: {trend_data['change_percentage']:.1f}%")
    
    # Get active alerts
    alerts = monitor.get_active_alerts(time_window=3600)
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts[:3]:  # Show first 3
        print(f"  [{alert['severity']}] {alert['message']}")
    
    # Get dashboard summary
    dashboard = monitor.get_dashboard_summary()
    print(f"\nDashboard Summary:")
    print(f"  Total metrics: {dashboard['total_metrics']}")
    print(f"  Active alerts: {dashboard['active_alerts']}")
    print(f"  Health status: {dashboard['health_status']}")


def example_full_integration():
    """
    Example 6: Full Integration
    
    Demonstrates using all Phase 2 features together with
    the BenchmarkDBAPIIPFS.
    """
    print("\n" + "="*60)
    print("Example 6: Full Integration (All Phases)")
    print("="*60)
    
    # Initialize configuration
    config = IPFSConfig(
        enable_ipfs_storage=True,
        enable_distributed=True,
        enable_knowledge_graph=True,
        enable_caching=True
    )
    
    # Initialize all components
    cache_manager = IPFSCacheManager(config)
    kg = BenchmarkKnowledgeGraph(config)
    populator = KnowledgeGraphPopulator(knowledge_graph=kg)
    optimizer = QueryOptimizer(cache_manager=cache_manager)
    monitor = PerformanceMonitor()
    executor = DistributedQueryExecutor(config=config, max_workers=4)
    
    # Initialize API with all features
    api = BenchmarkDBAPIIPFS(
        enable_ipfs=True,
        ipfs_config=config,
        enable_distributed=True,
        enable_knowledge_graph=True
    )
    
    print("✓ All components initialized")
    
    # Store benchmark result (Phase 2B)
    result = api.store_performance_result(
        model_name="bert-base",
        hardware_type="cuda",
        throughput=1250.5,
        latency=25.6,
        metadata={'framework': 'pytorch', 'batch_size': 32}
    )
    
    print(f"✓ Stored result (IPFS CID: {result.get('ipfs_cid', 'N/A')})")
    
    # Sync to IPFS (Phase 2B)
    if api.is_ipfs_available():
        db_cid = api.sync_to_ipfs()
        print(f"✓ Synced database (CID: {db_cid})")
    
    # Execute distributed query (Phase 2C)
    query = "SELECT * FROM benchmarks LIMIT 10"
    optimized = optimizer.optimize_query(query)
    print(f"✓ Query optimized ({len(optimized['optimizations_applied'])} optimizations)")
    
    # Record performance metric (Phase 2D)
    monitor.record_metric('query_time', 0.125)
    monitor.record_metric('result_count', 10)
    print("✓ Performance metrics recorded")
    
    # Get comprehensive status
    print(f"\nSystem Status:")
    print(f"  IPFS available: {api.is_ipfs_available()}")
    print(f"  Distributed enabled: {api.is_distributed_enabled()}")
    print(f"  Knowledge graph enabled: {api.is_knowledge_graph_enabled()}")
    print(f"  Health: {monitor._calculate_health_status()}")
    
    print("\n✓ Full integration demonstration complete!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Phase 2C & 2D Features - Comprehensive Examples")
    print("="*60)
    print("\nPhase 2C: Distributed Features")
    print("  - Distributed Query Execution")
    print("  - P2P Database Synchronization")
    print("  - Knowledge Graph Population")
    print("\nPhase 2D: Advanced Features")
    print("  - Query Optimization")
    print("  - Performance Monitoring")
    print("="*60)
    
    try:
        example_distributed_query_execution()
        example_p2p_synchronization()
        example_knowledge_graph_population()
        example_query_optimization()
        example_performance_monitoring()
        example_full_integration()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
