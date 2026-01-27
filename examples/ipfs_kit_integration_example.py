#!/usr/bin/env python3
"""
IPFS Kit Integration Example

This example demonstrates how to use the ipfs_kit_py integration
for distributed filesystem operations in the accelerate library.
"""

import sys
import json
from pathlib import Path

# Add ipfs_accelerate_py to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py import get_storage, reset_storage


def main():
    print("=" * 60)
    print("IPFS Kit Integration Example")
    print("=" * 60)
    print()
    
    # Get storage instance (automatically detects ipfs_kit_py availability)
    print("1. Initializing storage...")
    storage = get_storage()
    
    # Check backend status
    status = storage.get_backend_status()
    print(f"   - IPFS Kit available: {status['ipfs_kit_available']}")
    print(f"   - Using fallback: {status['using_fallback']}")
    print(f"   - Cache directory: {status['cache_dir']}")
    print()
    
    # Store some data
    print("2. Storing data...")
    data1 = b"Hello, IPFS Accelerate!"
    cid1 = storage.store(data1, filename="greeting.txt")
    print(f"   - Stored greeting with CID: {cid1}")
    
    data2 = json.dumps({
        "model": "bert-base-uncased",
        "task": "text-classification",
        "version": "1.0.0"
    }).encode()
    cid2 = storage.store(data2, filename="model_metadata.json", pin=True)
    print(f"   - Stored model metadata with CID: {cid2}")
    print()
    
    # Demonstrate content deduplication
    print("3. Demonstrating content deduplication...")
    data3 = b"Hello, IPFS Accelerate!"  # Same as data1
    cid3 = storage.store(data3, filename="greeting_copy.txt")
    print(f"   - Original CID: {cid1}")
    print(f"   - Copy CID: {cid3}")
    print(f"   - CIDs match: {cid1 == cid3} (same content = same CID)")
    print()
    
    # Retrieve data
    print("4. Retrieving data...")
    retrieved1 = storage.retrieve(cid1)
    print(f"   - Retrieved: {retrieved1.decode('utf-8')}")
    
    retrieved2 = storage.retrieve(cid2)
    metadata = json.loads(retrieved2.decode('utf-8'))
    print(f"   - Model metadata: {metadata}")
    print()
    
    # List all stored files
    print("5. Listing stored files...")
    files = storage.list_files()
    print(f"   - Total files: {len(files)}")
    for file_info in files:
        print(f"     • {file_info['filename']}")
        print(f"       CID: {file_info['cid']}")
        print(f"       Size: {file_info['size']} bytes")
        print(f"       Pinned: {file_info['pinned']}")
    print()
    
    # Demonstrate pinning
    print("6. Testing pinning functionality...")
    print(f"   - Pinning CID: {cid1}")
    storage.pin(cid1)
    print("   - Content pinned successfully")
    
    # Check existence
    print(f"   - Checking existence of CID: {cid1}")
    exists = storage.exists(cid1)
    print(f"   - Exists: {exists}")
    print()
    
    # Cleanup (optional)
    print("7. Cleanup...")
    print(f"   - Unpinning and deleting CID: {cid3}")
    storage.unpin(cid3)
    storage.delete(cid3)
    print("   - Cleanup complete")
    print()
    
    # Final status
    print("8. Final file count...")
    files = storage.list_files()
    print(f"   - Remaining files: {len(files)}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Automatic fallback (works with or without ipfs_kit_py)")
    print("  ✓ Content-addressed storage (CID generation)")
    print("  ✓ Content deduplication (same data = same CID)")
    print("  ✓ Pinning for persistence")
    print("  ✓ Metadata storage and retrieval")
    print("  ✓ File listing and management")
    print()
    
    if status['using_fallback']:
        print("Note: Running in fallback mode (local filesystem)")
        print("To enable distributed storage:")
        print("  1. Initialize the ipfs_kit_py submodule:")
        print("     git submodule update --init external/ipfs_kit_py")
        print("  2. Re-run this example")
    else:
        print("Note: Running with ipfs_kit_py integration enabled")
        print("Your data is using distributed storage!")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
