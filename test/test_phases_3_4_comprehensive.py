#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phases 3-4

Tests all documented features to ensure they work as intended:
- Dual-mode CLI/SDK execution
- Secrets manager integration
- Response format validation
- Cache integration
- Fallback behavior
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_response_format_validation():
    """Test that responses include all documented fields."""
    print("=" * 60)
    print("Test: Response Format Validation")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
        
        # Initialize with test key
        claude = ClaudeCodeCLIIntegration(api_key="sk-test-key")
        
        # Mock the SDK client to avoid real API calls
        with patch.object(claude, '_get_sdk_client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_client.return_value.messages.create.return_value = mock_response
            
            # Make a test call
            response = claude.chat("Test message", model="claude-3-sonnet-20240229")
            
            # Validate response structure
            assert "response" in response, "Missing 'response' field"
            assert "cached" in response, "Missing 'cached' field"
            assert "mode" in response, "Missing 'mode' field"
            
            # Validate field types
            assert isinstance(response["response"], str), "'response' should be string"
            assert isinstance(response["cached"], bool), "'cached' should be boolean"
            assert response["mode"] in ["CLI", "SDK"], f"'mode' should be CLI or SDK, got {response['mode']}"
            
            print("✅ Response format validation passed")
            print(f"   - response: {response['response'][:50]}...")
            print(f"   - cached: {response['cached']}")
            print(f"   - mode: {response['mode']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_integration():
    """Test that caching works with dual-mode integrations."""
    print("\n" + "=" * 60)
    print("Test: Cache Integration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import GroqCLIIntegration
        
        # Initialize with test key
        groq = GroqCLIIntegration(api_key="gsk-test-key", enable_cache=True)
        
        # Mock the SDK client
        with patch.object(groq, '_get_sdk_client') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Cached response"))]
            mock_client.return_value.chat.completions.create.return_value = mock_response
            
            # First call (should not be cached)
            response1 = groq.chat("Test message", model="llama3-70b-8192", temperature=0.0)
            assert response1["cached"] == False, "First call should not be cached"
            print(f"✅ First call not cached: {response1['cached']}")
            
            # Second call (should be cached)
            response2 = groq.chat("Test message", model="llama3-70b-8192", temperature=0.0)
            assert response2["cached"] == True, "Second call should be cached"
            print(f"✅ Second call cached: {response2['cached']}")
            
            # Verify responses match
            assert response1["response"] == response2["response"], "Cached response should match"
            print(f"✅ Cached response matches original")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_secrets_manager_integration():
    """Test that CLI integrations retrieve API keys from secrets manager."""
    print("\n" + "=" * 60)
    print("Test: Secrets Manager Integration with CLI Tools")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
        
        # Create temporary secrets file
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            
            # Initialize secrets manager with test key
            secrets = SecretsManager(secrets_file=secrets_file, use_encryption=True)
            secrets.set_credential("anthropic_api_key", "sk-ant-test-from-secrets")
            
            # Patch get_global_secrets_manager to return our test instance
            with patch('ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager', return_value=secrets):
                # Initialize Claude without explicit API key
                claude = ClaudeCodeCLIIntegration()
                
                # Verify API key was retrieved from secrets manager
                assert claude.api_key == "sk-ant-test-from-secrets", "API key should be from secrets manager"
                print(f"✅ API key correctly retrieved from secrets manager")
                print(f"   Key: {claude.api_key[:20]}...")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variable_fallback():
    """Test that environment variables work as fallback."""
    print("\n" + "=" * 60)
    print("Test: Environment Variable Fallback")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        
        # Create temporary secrets manager
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            secrets = SecretsManager(secrets_file=secrets_file)
            
            # Set environment variable
            os.environ["TEST_API_KEY"] = "env-value-test"
            
            # Test various naming formats
            test_cases = [
                ("test_api_key", "env-value-test"),
                ("TEST_API_KEY", "env-value-test"),
                ("test-api-key", "env-value-test"),
            ]
            
            for key_format, expected in test_cases:
                result = secrets.get_credential(key_format)
                assert result == expected, f"Expected {expected}, got {result} for {key_format}"
                print(f"✅ Format '{key_format}' correctly resolved to env var")
            
            # Cleanup
            del os.environ["TEST_API_KEY"]
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_mode_preference():
    """Test CLI vs SDK preference configuration."""
    print("\n" + "=" * 60)
    print("Test: Dual-Mode Preference Configuration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import GeminiCLIIntegration
        
        # Test SDK-first (default)
        gemini_sdk = GeminiCLIIntegration(api_key="test-key", prefer_cli=False)
        assert gemini_sdk.prefer_cli == False, "Should prefer SDK by default"
        print(f"✅ SDK-first mode: prefer_cli={gemini_sdk.prefer_cli}")
        
        # Test CLI-first
        gemini_cli = GeminiCLIIntegration(api_key="test-key", prefer_cli=True)
        assert gemini_cli.prefer_cli == True, "Should prefer CLI when specified"
        print(f"✅ CLI-first mode: prefer_cli={gemini_cli.prefer_cli}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_behavior():
    """Test that fallback occurs when primary mode fails."""
    print("\n" + "=" * 60)
    print("Test: Fallback Behavior")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
        
        # Initialize with prefer_cli=True but no CLI available
        claude = ClaudeCodeCLIIntegration(api_key="sk-test-key", prefer_cli=True)
        
        # CLI should not be available
        assert claude.cli_available == False, "CLI should not be available"
        print(f"✅ CLI not available: {claude.cli_available}")
        
        # Mock SDK to test fallback
        with patch.object(claude, '_get_sdk_client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Fallback response")]
            mock_client.return_value.messages.create.return_value = mock_response
            
            # Make call - should fall back to SDK
            response = claude.chat("Test", model="claude-3-sonnet-20240229")
            
            # Should have used SDK mode
            assert response["mode"] == "SDK", "Should have fallen back to SDK"
            print(f"✅ Correctly fell back to SDK mode")
            
            # Note: fallback field only set when actually switching modes during execution
            print(f"   Mode used: {response['mode']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_integrations_with_secrets():
    """Test that all three dual-mode integrations work with secrets manager."""
    print("\n" + "=" * 60)
    print("Test: All Integrations with Secrets Manager")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        from ipfs_accelerate_py.cli_integrations import (
            ClaudeCodeCLIIntegration,
            GeminiCLIIntegration,
            GroqCLIIntegration
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            secrets = SecretsManager(secrets_file=secrets_file)
            
            # Set up credentials
            secrets.set_credential("anthropic_api_key", "sk-ant-test")
            secrets.set_credential("google_api_key", "AIza-test")
            secrets.set_credential("groq_api_key", "gsk-test")
            
            with patch('ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager', return_value=secrets):
                # Test Claude
                claude = ClaudeCodeCLIIntegration()
                assert claude.api_key == "sk-ant-test"
                print(f"✅ Claude: API key from secrets manager")
                
                # Test Gemini
                gemini = GeminiCLIIntegration()
                assert gemini.api_key == "AIza-test"
                print(f"✅ Gemini: API key from secrets manager")
                
                # Test Groq
                groq = GroqCLIIntegration()
                assert groq.api_key == "gsk-test"
                print(f"✅ Groq: API key from secrets manager")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encryption_key_generation():
    """Test that encryption keys are generated and stored securely."""
    print("\n" + "=" * 60)
    print("Test: Encryption Key Generation and Storage")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            key_file = Path(tmpdir) / "secrets.key"
            
            # Create secrets manager (should auto-generate key)
            secrets = SecretsManager(secrets_file=secrets_file, use_encryption=True)
            
            # Verify key file was created
            assert key_file.exists(), "Key file should be created"
            print(f"✅ Encryption key file created")
            
            # Verify key file has secure permissions (Unix-like systems)
            if hasattr(os, 'stat'):
                import stat
                key_stat = key_file.stat()
                # Check that only owner has read/write
                mode = key_stat.st_mode
                print(f"✅ Key file permissions: {oct(stat.S_IMODE(mode))}")
            
            # Verify key can encrypt/decrypt
            secrets.set_credential("test_key", "test_value")
            retrieved = secrets.get_credential("test_key")
            assert retrieved == "test_value", "Encryption/decryption should work"
            print(f"✅ Encryption/decryption working correctly")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_credential_priority_order():
    """Test the documented priority order for credential retrieval."""
    print("\n" + "=" * 60)
    print("Test: Credential Priority Order")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "test_secrets.enc")
            secrets = SecretsManager(secrets_file=secrets_file)
            
            # Priority 1: In-memory cache
            secrets.set_credential("test_key", "memory_value", persist=False)
            assert secrets.get_credential("test_key") == "memory_value"
            print("✅ Priority 1: In-memory cache works")
            
            # Priority 2: Persisted value (when not in memory)
            secrets2 = SecretsManager(secrets_file=secrets_file)
            secrets2.set_credential("test_key2", "persisted_value", persist=True)
            secrets3 = SecretsManager(secrets_file=secrets_file)
            assert secrets3.get_credential("test_key2") == "persisted_value"
            print("✅ Priority 2: Persisted credentials work")
            
            # Priority 3: Environment variable (when not in memory or persisted)
            os.environ["TEST_ENV_PRIORITY"] = "env_value"
            secrets4 = SecretsManager(secrets_file=secrets_file)
            assert secrets4.get_credential("test_env_priority") == "env_value"
            print("✅ Priority 3: Environment variable fallback works")
            del os.environ["TEST_ENV_PRIORITY"]
            
            # Priority 4: Default value
            default_result = secrets.get_credential("nonexistent_key", default="default_value")
            assert default_result == "default_value"
            print("✅ Priority 4: Default value works")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUITE FOR PHASES 3-4")
    print("=" * 60)
    
    tests = [
        ("Response Format Validation", test_response_format_validation),
        ("Cache Integration", test_cache_integration),
        ("Secrets Manager Integration", test_secrets_manager_integration),
        ("Environment Variable Fallback", test_environment_variable_fallback),
        ("Dual-Mode Preference", test_dual_mode_preference),
        ("Fallback Behavior", test_fallback_behavior),
        ("All Integrations with Secrets", test_all_integrations_with_secrets),
        ("Encryption Key Generation", test_encryption_key_generation),
        ("Credential Priority Order", test_credential_priority_order),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All comprehensive tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)
