#!/usr/bin/env python3
"""
Test Phase 4: Secrets Manager

Tests the encrypted credential storage and API key management.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_secrets_manager_basic():
    """Test basic secrets manager functionality."""
    print("=" * 60)
    print("Testing Secrets Manager (Phase 4)")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        print("✅ Successfully imported SecretsManager")
        
        # Create temporary secrets file
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            
            print("\n" + "-" * 60)
            print("Test 1: Initialize secrets manager")
            print("-" * 60)
            
            manager = SecretsManager(
                secrets_file=secrets_file,
                use_encryption=True
            )
            print(f"✅ Secrets manager initialized with file: {secrets_file}")
            
            print("\n" + "-" * 60)
            print("Test 2: Set and get credentials")
            print("-" * 60)
            
            manager.set_credential("test_api_key", "sk-test-12345")
            retrieved = manager.get_credential("test_api_key")
            
            if retrieved == "sk-test-12345":
                print("✅ Credential stored and retrieved successfully")
            else:
                print(f"❌ Expected 'sk-test-12345', got '{retrieved}'")
            
            print("\n" + "-" * 60)
            print("Test 3: List credential keys")
            print("-" * 60)
            
            manager.set_credential("openai_api_key", "sk-openai-test")
            manager.set_credential("anthropic_api_key", "sk-anthropic-test")
            
            keys = manager.list_credential_keys()
            print(f"✅ Found {len(keys)} credentials: {keys}")
            
            print("\n" + "-" * 60)
            print("Test 4: Persistence (save and reload)")
            print("-" * 60)
            
            # Create new manager instance to test persistence
            manager2 = SecretsManager(secrets_file=secrets_file)
            retrieved2 = manager2.get_credential("openai_api_key")
            
            if retrieved2 == "sk-openai-test":
                print("✅ Credentials persisted and reloaded successfully")
            else:
                print(f"❌ Expected 'sk-openai-test', got '{retrieved2}'")
            
            print("\n" + "-" * 60)
            print("Test 5: Environment variable fallback")
            print("-" * 60)
            
            os.environ["TEST_ENV_KEY"] = "env-value-test"
            retrieved3 = manager.get_credential("test_env_key")
            
            if retrieved3 == "env-value-test":
                print("✅ Environment variable fallback working")
            else:
                print(f"❌ Expected 'env-value-test', got '{retrieved3}'")
            
            del os.environ["TEST_ENV_KEY"]
            
            print("\n" + "-" * 60)
            print("Test 6: Delete credential")
            print("-" * 60)
            
            manager.delete_credential("test_api_key")
            retrieved4 = manager.get_credential("test_api_key")
            
            if retrieved4 is None:
                print("✅ Credential deleted successfully")
            else:
                print(f"❌ Expected None, got '{retrieved4}'")
            
            print("\n" + "-" * 60)
            print("Test 7: Encryption verification")
            print("-" * 60)
            
            # Read raw file to verify it's encrypted
            with open(secrets_file, 'rb') as f:
                raw_data = f.read()
            
            # Should not contain plaintext credentials
            if b"sk-openai-test" not in raw_data:
                print("✅ Credentials are encrypted (not found in plaintext)")
            else:
                print("❌ Credentials found in plaintext - encryption may not be working")
            
        print("\n" + "=" * 60)
        print("Secrets Manager Tests Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_secrets_manager_no_encryption():
    """Test secrets manager without encryption."""
    print("\n" + "=" * 60)
    print("Testing Secrets Manager without Encryption")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets_plain.json")
            
            manager = SecretsManager(
                secrets_file=secrets_file,
                use_encryption=False
            )
            print("✅ Secrets manager initialized without encryption")
            
            manager.set_credential("test_key", "test_value")
            retrieved = manager.get_credential("test_key")
            
            if retrieved == "test_value":
                print("✅ Unencrypted storage working correctly")
            else:
                print(f"❌ Expected 'test_value', got '{retrieved}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_global_secrets_manager():
    """Test global secrets manager instance."""
    print("\n" + "=" * 60)
    print("Testing Global Secrets Manager Instance")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager
        
        manager1 = get_global_secrets_manager()
        manager2 = get_global_secrets_manager()
        
        if manager1 is manager2:
            print("✅ Global instance working correctly (same object returned)")
        else:
            print("❌ Different objects returned for global instance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = True
    
    success = test_secrets_manager_basic() and success
    success = test_secrets_manager_no_encryption() and success
    success = test_global_secrets_manager() and success
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
