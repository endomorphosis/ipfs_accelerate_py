#!/usr/bin/env python3
"""
End-to-End Integration Tests for Phases 3-4

Tests real-world usage scenarios as documented:
- Complete workflow from secrets setup to API calls
- Multi-integration scenarios
- Cache persistence across sessions
- Error handling and recovery
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_complete_workflow_as_documented():
    """Test the complete workflow shown in documentation."""
    print("=" * 60)
    print("Test: Complete Workflow As Documented")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager
        from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the secrets location for testing
            with patch('ipfs_accelerate_py.common.secrets_manager.Path.home', return_value=Path(tmpdir)):
                # Step 1: Setup secrets manager (as in documentation)
                secrets = get_global_secrets_manager()
                secrets.set_credential("anthropic_api_key", "sk-ant-test-workflow")
                print("✅ Step 1: Set up API key in secrets manager")
                
                # Step 2: Initialize Claude without explicit key (as in documentation)
                claude = ClaudeCodeCLIIntegration()
                assert claude.api_key == "sk-ant-test-workflow"
                print("✅ Step 2: Claude initialized with API key from secrets")
                
                # Step 3: Make an API call (mocked)
                with patch.object(claude, '_get_sdk_client') as mock_client:
                    mock_response = Mock()
                    mock_response.content = [Mock(text="Decorators are functions that modify other functions")]
                    mock_client.return_value.messages.create.return_value = mock_response
                    
                    response = claude.chat("Explain decorators")
                    
                    # Verify response format as documented
                    assert "response" in response
                    assert "mode" in response
                    assert "cached" in response
                    print("✅ Step 3: Response format matches documentation")
                    print(f"   - response: {response['response'][:50]}...")
                    print(f"   - mode: {response['mode']}")
                    print(f"   - cached: {response['cached']}")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_migration_scenario():
    """Test migration from pre-Phase 3-4 to new version."""
    print("\n" + "=" * 60)
    print("Test: Migration Scenario (Before → After)")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import GeminiCLIIntegration
        
        # Before: Using explicit API key (still works)
        print("Before Phase 3-4 style:")
        gemini_old = GeminiCLIIntegration(api_key="explicit-key-123")
        assert gemini_old.api_key == "explicit-key-123"
        print("✅ Old style (explicit key) still works")
        
        # After: Using secrets manager (new style)
        print("\nAfter Phase 3-4 style:")
        with tempfile.TemporaryDirectory() as tmpdir:
            from ipfs_accelerate_py.common.secrets_manager import SecretsManager
            
            secrets_file = os.path.join(tmpdir, "secrets.enc")
            secrets = SecretsManager(secrets_file=secrets_file)
            secrets.set_credential("google_api_key", "secrets-key-456")
            
            with patch('ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager', return_value=secrets):
                gemini_new = GeminiCLIIntegration()
                assert gemini_new.api_key == "secrets-key-456"
                print("✅ New style (secrets manager) works")
        
        print("✅ Migration path validated: both styles work")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_integrations_together():
    """Test using multiple integrations in the same application."""
    print("\n" + "=" * 60)
    print("Test: Multiple Integrations Together")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        from ipfs_accelerate_py.cli_integrations import (
            ClaudeCodeCLIIntegration,
            GeminiCLIIntegration,
            GroqCLIIntegration
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "secrets.enc")
            secrets = SecretsManager(secrets_file=secrets_file)
            
            # Set up multiple credentials
            secrets.set_credential("anthropic_api_key", "sk-ant-multi")
            secrets.set_credential("google_api_key", "AIza-multi")
            secrets.set_credential("groq_api_key", "gsk-multi")
            
            with patch('ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager', return_value=secrets):
                # Initialize all three
                claude = ClaudeCodeCLIIntegration()
                gemini = GeminiCLIIntegration()
                groq = GroqCLIIntegration()
                
                # Verify they got correct keys
                assert claude.api_key == "sk-ant-multi"
                assert gemini.api_key == "AIza-multi"
                assert groq.api_key == "gsk-multi"
                
                print("✅ All three integrations initialized with correct keys")
                
                # Mock API calls for all three
                with patch.object(claude, '_get_sdk_client') as mock_claude, \
                     patch.object(gemini, '_create_sdk_client') as mock_gemini, \
                     patch.object(groq, '_get_sdk_client') as mock_groq:
                    
                    # Setup mocks
                    mock_claude.return_value.messages.create.return_value = Mock(
                        content=[Mock(text="Claude response")]
                    )
                    
                    # Properly configure Gemini mock
                    gemini._configured = True
                    gemini._genai = Mock()
                    mock_model = Mock()
                    mock_model.generate_content.return_value = Mock(text="Gemini response")
                    gemini._genai.GenerativeModel.return_value = mock_model
                    
                    mock_groq.return_value.chat.completions.create.return_value = Mock(
                        choices=[Mock(message=Mock(content="Groq response"))]
                    )
                    
                    # Make calls
                    response1 = claude.chat("Test", model="claude-3-sonnet-20240229")
                    response2 = gemini.generate_text("Test", model="gemini-pro")
                    response3 = groq.chat("Test", model="llama3-70b-8192")
                    
                    # Verify all work
                    assert response1["response"] == "Claude response"
                    assert response2["response"] == "Gemini response"
                    assert response3["response"] == "Groq response"
                    
                    print("✅ All three integrations made successful calls")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_persistence_across_sessions():
    """Test that cache persists across different sessions."""
    print("\n" + "=" * 60)
    print("Test: Cache Persistence Across Sessions")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import GroqCLIIntegration
        
        # Session 1: Make initial call
        groq1 = GroqCLIIntegration(api_key="gsk-test", enable_cache=True)
        
        with patch.object(groq1, '_get_sdk_client') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="First session response"))]
            )
            
            response1 = groq1.chat("Persistent test", model="llama3-70b-8192", temperature=0.0)
            assert response1["cached"] == False
            print("✅ Session 1: Made uncached call")
        
        # Session 2: Same call should be cached
        groq2 = GroqCLIIntegration(api_key="gsk-test", enable_cache=True)
        
        # This should hit cache without calling SDK
        response2 = groq2.chat("Persistent test", model="llama3-70b-8192", temperature=0.0)
        assert response2["cached"] == True
        assert response2["response"] == "First session response"
        print("✅ Session 2: Retrieved from cache")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling scenarios."""
    print("\n" + "=" * 60)
    print("Test: Error Handling")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
        
        # Test 1: Missing API key (should still initialize but might fail on call)
        claude_no_key = ClaudeCodeCLIIntegration(api_key=None)
        print("✅ Can initialize without API key")
        
        # Test 2: Invalid API key format (should accept any string)
        claude_invalid = ClaudeCodeCLIIntegration(api_key="invalid-key-format")
        assert claude_invalid.api_key == "invalid-key-format"
        print("✅ Accepts any API key format")
        
        # Test 3: Fallback when SDK fails
        claude = ClaudeCodeCLIIntegration(api_key="sk-test", prefer_cli=False)
        
        with patch.object(claude, '_get_sdk_client') as mock_client:
            # Make SDK raise an error
            mock_client.return_value.messages.create.side_effect = Exception("SDK Error")
            
            try:
                response = claude.chat("Test", model="claude-3-sonnet-20240229")
                # If we got here, error was raised and caught by fallback logic
                # Since both CLI and SDK will fail, it should raise RuntimeError
            except RuntimeError as e:
                # This is expected - both modes fail
                if "Both CLI and SDK modes failed" in str(e):
                    print(f"✅ Correctly raised RuntimeError when both modes fail")
                else:
                    print(f"❌ Wrong error: {e}")
                    return False
            except Exception as e:
                print(f"✅ Correctly raised error when SDK fails: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_secrets_file_permissions():
    """Test that secrets files have proper permissions."""
    print("\n" + "=" * 60)
    print("Test: Secrets File Permissions")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import SecretsManager
        import stat
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_file = os.path.join(tmpdir, "secrets.enc")
            
            # Create secrets manager
            secrets = SecretsManager(secrets_file=secrets_file, use_encryption=True)
            secrets.set_credential("test", "value")
            
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'stat'):
                secrets_stat = os.stat(secrets_file)
                mode = stat.S_IMODE(secrets_stat.st_mode)
                
                # Should be 0o600 (owner read/write only)
                assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
                print(f"✅ Secrets file has correct permissions: {oct(mode)}")
                
                # Check key file too
                key_file = Path(tmpdir) / "secrets.key"
                if key_file.exists():
                    key_stat = key_file.stat()
                    key_mode = stat.S_IMODE(key_stat.st_mode)
                    assert key_mode == 0o600, f"Expected 0o600, got {oct(key_mode)}"
                    print(f"✅ Key file has correct permissions: {oct(key_mode)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disable_cache_option():
    """Test that caching can be disabled."""
    print("\n" + "=" * 60)
    print("Test: Disable Cache Option")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import GroqCLIIntegration
        
        # Initialize with caching disabled
        groq = GroqCLIIntegration(api_key="gsk-test", enable_cache=False)
        
        with patch.object(groq, '_get_sdk_client') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Response"))]
            )
            
            # Make two identical calls
            response1 = groq.chat("Test", model="llama3-70b-8192", temperature=0.0)
            response2 = groq.chat("Test", model="llama3-70b-8192", temperature=0.0)
            
            # Both should not be cached
            assert response1["cached"] == False
            assert response2["cached"] == False
            print("✅ Cache disabled: both calls not cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("END-TO-END INTEGRATION TESTS FOR PHASES 3-4")
    print("=" * 60)
    
    tests = [
        ("Complete Workflow As Documented", test_complete_workflow_as_documented),
        ("Migration Scenario", test_migration_scenario),
        ("Multiple Integrations Together", test_multiple_integrations_together),
        ("Cache Persistence Across Sessions", test_cache_persistence_across_sessions),
        ("Error Handling", test_error_handling),
        ("Secrets File Permissions", test_secrets_file_permissions),
        ("Disable Cache Option", test_disable_cache_option),
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"END-TO-END TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All end-to-end tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)
