#!/usr/bin/env python
"""
Example demonstrating IPFS-enhanced BenchmarkDBAPI usage.

This example shows how to:
1. Enable IPFS storage for benchmark databases
2. Store benchmark results with automatic IPFS backup
3. Sync database to IPFS
4. Query using distributed operations
5. Use knowledge graph for similarity search

Usage:
    python example_ipfs_benchmark_api.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from data.duckdb.core.benchmark_db_api_ipfs import BenchmarkDBAPIIPFS, create_benchmark_api
    from data.duckdb.ipfs_integration import IPFSConfig
except ImportError as e:
    print(f"Error: Could not import modules: {e}")
    print("\nMake sure:")
    print("1. You're in the correct directory")
    print("2. ipfs_datasets_py and ipfs_kit_py submodules are initialized")
    sys.exit(1)


def example_basic_usage():
    """Example 1: Basic usage without IPFS (backward compatible)."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage (IPFS Disabled)")
    print("="*60)
    
    # Create API without IPFS
    api = create_benchmark_api(
        db_path="./example_benchmark.duckdb",
        enable_ipfs=False
    )
    
    # Store a performance result
    result = api.store_performance_result(
        model_name="bert-base-uncased",
        hardware_type="cuda",
        device_name="NVIDIA RTX 3090",
        batch_size=32,
        precision="fp16",
        test_case="inference",
        throughput=1250.5,
        latency_avg=25.6,
        memory_peak=8192.0
    )
    
    print(f"✓ Stored result: {result['result_id']}")
    print(f"  Database: {api.db_path}")
    print(f"  IPFS: {result.get('ipfs_cid', 'Not enabled')}")


def example_ipfs_enabled():
    """Example 2: Enable IPFS storage."""
    print("\n" + "="*60)
    print("Example 2: IPFS-Enabled Storage")
    print("="*60)
    
    # Create API with IPFS enabled
    api = create_benchmark_api(
        db_path="./example_ipfs_benchmark.duckdb",
        enable_ipfs=True
    )
    
    # Check IPFS availability
    status = api.get_status()
    print(f"IPFS Available: {status['ipfs']['backend_available']}")
    
    if not status['ipfs']['backend_available']:
        print("⚠️  IPFS not available. Results will be stored locally only.")
        print("   To enable IPFS:")
        print("   1. Initialize submodules: git submodule update --init")
        print("   2. Ensure IPFS daemon is running (if using local IPFS)")
        return
    
    # Store a result - automatically backed up to IPFS if available
    result = api.store_performance_result(
        model_name="gpt2",
        hardware_type="cuda",
        device_name="NVIDIA A100",
        batch_size=16,
        precision="fp32",
        test_case="generation",
        throughput=850.3,
        latency_avg=18.7,
        memory_peak=16384.0
    )
    
    print(f"✓ Stored result: {result['result_id']}")
    print(f"  IPFS CID: {result.get('ipfs_cid', 'N/A')}")
    
    # Sync entire database to IPFS
    db_cid = api.sync_to_ipfs()
    if db_cid:
        print(f"✓ Database synced to IPFS: {db_cid}")
        print(f"  Can be retrieved from any IPFS node!")
    else:
        print("⚠️  Database sync to IPFS failed or not available")


def example_full_features():
    """Example 3: All IPFS features enabled."""
    print("\n" + "="*60)
    print("Example 3: Full IPFS Integration")
    print("="*60)
    
    # Custom configuration
    config = IPFSConfig(
        enable_ipfs_storage=True,
        enable_distributed=True,
        enable_knowledge_graph=True,
        enable_cache=True
    )
    
    # Create API with all features
    api = BenchmarkDBAPIIPFS(
        db_path="./example_full_benchmark.duckdb",
        enable_ipfs=True,
        ipfs_config=config,
        enable_distributed=True,
        enable_knowledge_graph=True
    )
    
    # Get comprehensive status
    status = api.get_status()
    print("\nFeature Status:")
    print(f"  IPFS Storage: {status['ipfs']['enabled']}")
    print(f"  Distributed Ops: {status['distributed']['enabled']}")
    print(f"  Knowledge Graph: {status['knowledge_graph']['enabled']}")
    
    # Store multiple results
    models = [
        ("resnet50", "cuda", 2048.5, 12.3),
        ("efficientnet-b0", "cuda", 3250.8, 9.7),
        ("mobilenet-v2", "cuda", 4100.2, 7.8),
    ]
    
    for model_name, hw_type, throughput, latency in models:
        result = api.store_performance_result(
            model_name=model_name,
            hardware_type=hw_type,
            device_name="NVIDIA RTX 3090",
            batch_size=64,
            precision="fp16",
            test_case="inference",
            throughput=throughput,
            latency_avg=latency
        )
        print(f"✓ Stored {model_name}: {result['result_id']}")
    
    # Search for similar benchmarks using knowledge graph
    if status['knowledge_graph']['enabled']:
        print("\nSearching for similar benchmarks to 'resnet50'...")
        similar = api.search_similar_benchmarks("resnet50", top_k=2)
        if similar:
            for item in similar:
                print(f"  - {item}")
        else:
            print("  (Knowledge graph needs more data)")
    
    # Sync to IPFS
    db_cid = api.sync_to_ipfs()
    if db_cid:
        print(f"\n✓ Full database on IPFS: {db_cid}")
        
        # Get version history
        versions = api.get_version_history()
        print(f"  Version history: {len(versions)} version(s)")
        for v in versions:
            print(f"    - {v['cid']} ({v['timestamp']})")


def example_migration():
    """Example 4: Migrate existing database to IPFS."""
    print("\n" + "="*60)
    print("Example 4: Database Migration")
    print("="*60)
    
    # Create a regular database
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    
    db_path = "./example_migrate.duckdb"
    print(f"Creating test database: {db_path}")
    
    api = BenchmarkDBAPI(db_path=db_path)
    
    # Add some data
    for i in range(3):
        api.store_performance_result(
            model_name=f"model-{i}",
            hardware_type="cpu",
            throughput=100.0 * (i + 1),
            latency_avg=10.0 / (i + 1)
        )
    
    print(f"✓ Created database with 3 results")
    
    # Now migrate to IPFS
    print("\nMigrating to IPFS...")
    
    from data.duckdb.ipfs_integration import IPFSDBMigration, IPFSConfig
    
    config = IPFSConfig(enable_ipfs_storage=True)
    migration = IPFSDBMigration(config=config)
    
    cid = migration.migrate_database(db_path, create_backup=True)
    
    if cid:
        print(f"✓ Database migrated to IPFS: {cid}")
        print(f"  Backup created: {db_path}.pre-ipfs-backup")
        print(f"  Metadata saved: .{Path(db_path).name}.ipfs.json")
    else:
        print("⚠️  Migration failed or IPFS not available")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("IPFS-Enhanced Benchmark Database API - Examples")
    print("="*70)
    
    try:
        # Run examples
        example_basic_usage()
        example_ipfs_enabled()
        example_full_features()
        example_migration()
        
        print("\n" + "="*70)
        print("Examples completed!")
        print("="*70)
        print("\nNext steps:")
        print("1. Check the created database files (*.duckdb)")
        print("2. Review IPFS metadata files (.*ipfs.json)")
        print("3. Try the migration script: python migrate_to_ipfs.py")
        print("4. Explore the API documentation")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
