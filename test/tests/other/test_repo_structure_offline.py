#!/usr/bin/env python3
"""
Offline test script for HuggingFace repository structure functionality.
This tests the implementation without requiring internet access.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import ipfs_accelerate_py
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py.model_manager import (
    ModelManager, 
    ModelMetadata,
    ModelType,
    DataType,
    IOSpec,
    create_model_from_huggingface,
    get_file_hash_from_structure,
    list_files_by_extension
)

def create_mock_repo_structure(model_id: str) -> dict:
    """Create a mock repository structure for testing."""
    return {
        "model_id": model_id,
        "branch": "main", 
        "fetched_at": datetime.now().isoformat(),
        "files": {
            "config.json": {
                "size": 665,
                "lfs": {},
                "oid": "6e3c55a11b8e2e30a4fdbee5b1fb8e28c2c4b8f0",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/config.json"
            },
            "pytorch_model.bin": {
                "size": 665,
                "lfs": {
                    "size": 503382240,
                    "sha256": "7cb18dc9bafbfcf74629a4b760af1b160957a83e",
                    "pointer_size": 135
                },
                "oid": "7cb18dc9bafbfcf74629a4b760af1b160957a83e",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin"
            },
            "tokenizer.json": {
                "size": 1356917,
                "lfs": {},
                "oid": "b70400ec13e62b577e6ac83a7e8c176f923b0e6d",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/tokenizer.json"
            },
            "vocab.json": {
                "size": 1042301,
                "lfs": {},
                "oid": "f9c4d7b1d57c84f81e6bc6dfcb12c32f0a3f5e2d",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/vocab.json"
            },
            "merges.txt": {
                "size": 456318,
                "lfs": {},
                "oid": "e7dc4e0e67f9c4df66d3d5b8ce40b5a7e0b1e234",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/merges.txt"
            },
            "README.md": {
                "size": 4321,
                "lfs": {},
                "oid": "9e7c4b5b8fb9c1234567890abcdef1234567890a",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/README.md"
            },
            "training_args.json": {
                "size": 1543,
                "lfs": {},
                "oid": "a1b2c3d4e5f67890abcdef1234567890abcdef12",
                "download_url": f"https://huggingface.co/{model_id}/resolve/main/training_args.json"
            }
        },
        "total_files": 7,
        "total_size": 2462665
    }

def test_repo_structure_offline():
    """Test repository structure functionality with mock data."""
    print("Testing HuggingFace repository structure functionality (offline)...")
    
    model_id = "gpt2"
    
    print(f"\n1. Testing utility functions with mock data")
    
    # Create mock repository structure
    repo_structure = create_mock_repo_structure(model_id)
    print(f"✅ Created mock repository structure")
    print(f"   - Total files: {repo_structure.get('total_files', 0)}")
    print(f"   - Total size: {repo_structure.get('total_size', 0):,} bytes")
    
    # Test utility functions
    print(f"\n2. Testing repository utility functions")
    
    # Test getting file hash
    config_hash = get_file_hash_from_structure(repo_structure, "config.json")
    print(f"   - config.json hash: {config_hash}")
    assert config_hash == "6e3c55a11b8e2e30a4fdbee5b1fb8e28c2c4b8f0"
    
    # Test non-existent file
    missing_hash = get_file_hash_from_structure(repo_structure, "nonexistent.txt")
    print(f"   - nonexistent.txt hash: {missing_hash}")
    assert missing_hash is None
    
    # Test listing files by extension
    json_files = list_files_by_extension(repo_structure, ".json")
    print(f"   - JSON files: {json_files}")
    assert "config.json" in json_files
    assert "tokenizer.json" in json_files
    
    txt_files = list_files_by_extension(repo_structure, ".txt")
    print(f"   - Text files: {txt_files}")
    assert "merges.txt" in txt_files
    
    md_files = list_files_by_extension(repo_structure, ".md")
    print(f"   - Markdown files: {md_files}")
    assert "README.md" in md_files
    
    print(f"\n3. Testing ModelManager integration")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "test_models.json")
        
        # Create model manager
        manager = ModelManager(storage_path=storage_path, use_database=False)
        
        # Create model metadata with mock repository structure
        print(f"   - Creating model metadata with repository structure...")
        
        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name="GPT-2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="GPT2LMHeadModel",
            inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            huggingface_config={
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "vocab_size": 50257
            },
            repository_structure=repo_structure,
            source_url=f"https://huggingface.co/{model_id}"
        )
        
        # Add to manager
        success = manager.add_model(model_metadata)
        assert success
        print(f"   ✅ Successfully added model to manager")
        
        # Test repository structure queries
        file_hash = manager.get_model_file_hash(model_id, "config.json")
        print(f"   - File hash for config.json: {file_hash}")
        assert file_hash == "6e3c55a11b8e2e30a4fdbee5b1fb8e28c2c4b8f0"
        
        # Test finding models with specific files
        models_with_config = manager.get_models_with_file("config.json")
        print(f"   - Models with config.json: {len(models_with_config)}")
        assert len(models_with_config) == 1
        assert models_with_config[0].model_id == model_id
        
        models_with_bin = manager.get_models_with_file("pytorch_model.bin")
        print(f"   - Models with pytorch_model.bin: {len(models_with_bin)}")
        assert len(models_with_bin) == 1
        
        models_with_fake = manager.get_models_with_file("nonexistent.file")
        print(f"   - Models with nonexistent.file: {len(models_with_fake)}")
        assert len(models_with_fake) == 0
        
        # Test statistics
        stats = manager.get_stats()
        print(f"   - Models with repository structure: {stats.get('models_with_repo_structure', 0)}")
        print(f"   - Total tracked files: {stats.get('total_tracked_files', 0)}")
        assert stats.get('models_with_repo_structure', 0) == 1
        assert stats.get('total_tracked_files', 0) == 7
        
        # Test persistence - save and reload
        print(f"   - Testing persistence...")
        manager._save_data()
        
        # Create new manager instance and load data
        manager2 = ModelManager(storage_path=storage_path, use_database=False)
        loaded_model = manager2.get_model(model_id)
        assert loaded_model is not None
        assert loaded_model.repository_structure is not None
        assert loaded_model.repository_structure["total_files"] == 7
        print(f"   ✅ Successfully loaded repository structure from storage")
        
        # Test refresh functionality (may succeed if internet available, should fail gracefully without)
        print(f"   - Testing repository structure refresh (should fail gracefully)...")
        refresh_success = manager.refresh_repository_structure(model_id)
        print(f"   - Refresh result: {refresh_success} (depends on internet availability)")
        # Test passes whether refresh succeeds or fails - the important thing is it doesn't crash
        assert isinstance(refresh_success, bool), "Refresh should return a boolean"
    
    print(f"\n4. Testing create_model_from_huggingface without fetching")
    
    # Test creating model without repository structure fetching
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": 50257
    }
    
    model_without_repo = create_model_from_huggingface(
        model_id="test-model",
        hf_config=hf_config,
        fetch_repo_structure=False
    )
    
    assert model_without_repo.repository_structure is None
    print(f"   ✅ Successfully created model without repository structure")
    
    # Test creating model with repository structure fetching (will fail gracefully)
    model_with_repo_attempt = create_model_from_huggingface(
        model_id="test-model-2",
        hf_config=hf_config,
        fetch_repo_structure=True
    )
    
    # Should fail gracefully and return None for repository_structure
    assert model_with_repo_attempt.repository_structure is None
    print(f"   ✅ Gracefully handled repository structure fetching failure")
    
    print(f"\n✅ All offline repository structure tests completed successfully!")
    # Test passes if we get here without exceptions

if __name__ == "__main__":
    try:
        test_repo_structure_offline()
        success = True
    except Exception as e:
        print(f"Test failed: {e}")
        success = False
    sys.exit(0 if success else 1)