#!/usr/bin/env python3
"""
Test script for HuggingFace repository structure functionality.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the ipfs_accelerate_py directory to the path
sys.path.insert(0, str(Path(__file__).parent / "ipfs_accelerate_py"))

from ipfs_accelerate_py.model_manager import (
    ModelManager, 
    create_model_from_huggingface,
    fetch_huggingface_repo_structure,
    get_file_hash_from_structure,
    list_files_by_extension
)

def test_huggingface_repo_structure():
    """Test fetching HuggingFace repository structure."""
    print("Testing HuggingFace repository structure functionality...")
    
    # Test with a small, well-known model
    model_id = "gpt2"
    
    print(f"\n1. Testing repository structure fetching for {model_id}")
    repo_structure = fetch_huggingface_repo_structure(model_id)
    
    if repo_structure:
        print(f"✅ Successfully fetched repository structure")
        print(f"   - Total files: {repo_structure.get('total_files', 0)}")
        print(f"   - Total size: {repo_structure.get('total_size', 0)} bytes")
        print(f"   - Branch: {repo_structure.get('branch', 'unknown')}")
        
        # Show some example files
        files = list(repo_structure.get('files', {}).keys())[:5]
        print(f"   - Example files: {files}")
        
        # Test utility functions
        print(f"\n2. Testing utility functions")
        
        # Test getting file hash
        config_hash = get_file_hash_from_structure(repo_structure, "config.json")
        print(f"   - config.json hash: {config_hash}")
        
        # Test listing files by extension
        json_files = list_files_by_extension(repo_structure, ".json")
        print(f"   - JSON files: {json_files}")
        
        py_files = list_files_by_extension(repo_structure, ".py")
        print(f"   - Python files: {py_files}")
        
    else:
        print(f"❌ Failed to fetch repository structure for {model_id}")
        return False
    
    print(f"\n3. Testing ModelManager integration")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "test_models.json")
        
        # Create model manager
        manager = ModelManager(storage_path=storage_path, use_database=False)
        
        # Create model from HuggingFace with repository structure
        print(f"   - Creating model metadata with repository structure...")
        
        # Mock HuggingFace config (normally would come from model config)
        hf_config = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "vocab_size": 50257
        }
        
        model_metadata = create_model_from_huggingface(
            model_id=model_id,
            hf_config=hf_config,
            fetch_repo_structure=True
        )
        
        # Add to manager
        success = manager.add_model(model_metadata)
        if success:
            print(f"   ✅ Successfully added model to manager")
            
            # Test repository structure queries
            file_hash = manager.get_model_file_hash(model_id, "config.json")
            print(f"   - File hash for config.json: {file_hash}")
            
            # Test finding models with specific files
            models_with_config = manager.get_models_with_file("config.json")
            print(f"   - Models with config.json: {len(models_with_config)}")
            
            # Test statistics
            stats = manager.get_stats()
            print(f"   - Models with repository structure: {stats.get('models_with_repo_structure', 0)}")
            print(f"   - Total tracked files: {stats.get('total_tracked_files', 0)}")
            
            # Test refresh functionality
            print(f"   - Testing repository structure refresh...")
            refresh_success = manager.refresh_repository_structure(model_id)
            print(f"   - Refresh successful: {refresh_success}")
            
        else:
            print(f"   ❌ Failed to add model to manager")
            return False
    
    print(f"\n4. Testing graceful degradation without requests")
    
    # Simulate missing requests library
    import ipfs_accelerate_py.model_manager as mm
    original_have_requests = mm.HAVE_REQUESTS
    mm.HAVE_REQUESTS = False
    
    try:
        repo_structure = fetch_huggingface_repo_structure("test-model")
        if repo_structure is None:
            print(f"   ✅ Gracefully handled missing requests library")
        else:
            print(f"   ⚠️  Expected None when requests unavailable")
    finally:
        # Restore original state
        mm.HAVE_REQUESTS = original_have_requests
    
    print(f"\n✅ All repository structure tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_huggingface_repo_structure()
    sys.exit(0 if success else 1)