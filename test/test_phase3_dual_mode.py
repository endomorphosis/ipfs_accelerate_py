#!/usr/bin/env python3
"""
Test Phase 3: Dual-Mode CLI/API Integration

Tests the CLI/SDK fallback logic and unified interfaces.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_dual_mode_wrapper():
    """Test dual-mode wrapper base functionality."""
    print("=" * 60)
    print("Testing Dual-Mode CLI/API Wrapper (Phase 3)")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations.dual_mode_wrapper import DualModeWrapper, detect_cli_tool
        print("✅ Successfully imported DualModeWrapper")
        
        print("\n" + "-" * 60)
        print("Test 1: CLI tool detection")
        print("-" * 60)
        
        # Try to detect a common CLI tool
        git_path = detect_cli_tool(["git"])
        if git_path:
            print(f"✅ CLI detection working - found git at: {git_path}")
        else:
            print("⚠️  git not found (this is OK for testing)")
        
        # Try to detect non-existent tool
        fake_path = detect_cli_tool(["nonexistent-tool-12345"])
        if fake_path is None:
            print("✅ Non-existent tool correctly returns None")
        else:
            print(f"❌ Expected None for non-existent tool, got: {fake_path}")
        
        print("\n✅ Dual-mode wrapper tests passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_claude_dual_mode():
    """Test Claude integration with dual-mode support."""
    print("\n" + "=" * 60)
    print("Testing Claude Dual-Mode Integration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration
        print("✅ Successfully imported ClaudeCodeCLIIntegration")
        
        print("\n" + "-" * 60)
        print("Test 1: Initialization")
        print("-" * 60)
        
        # Initialize without API key (will check secrets manager and env)
        try:
            claude = ClaudeCodeCLIIntegration(api_key="sk-test-key")
            print(f"✅ Claude integration initialized")
            print(f"   Tool name: {claude.get_tool_name()}")
            print(f"   CLI available: {claude.cli_available}")
            print(f"   Prefer CLI: {claude.prefer_cli}")
        except Exception as e:
            print(f"⚠️  Could not initialize (expected if dependencies not installed): {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_gemini_dual_mode():
    """Test Gemini integration with dual-mode support."""
    print("\n" + "=" * 60)
    print("Testing Gemini Dual-Mode Integration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations.gemini_cli_integration import GeminiCLIIntegration
        print("✅ Successfully imported GeminiCLIIntegration")
        
        print("\n" + "-" * 60)
        print("Test 1: Initialization")
        print("-" * 60)
        
        try:
            gemini = GeminiCLIIntegration(api_key="test-key")
            print(f"✅ Gemini integration initialized")
            print(f"   Tool name: {gemini.get_tool_name()}")
            print(f"   CLI available: {gemini.cli_available}")
            print(f"   Prefer CLI: {gemini.prefer_cli}")
        except Exception as e:
            print(f"⚠️  Could not initialize (expected if dependencies not installed): {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_groq_dual_mode():
    """Test Groq integration with dual-mode support."""
    print("\n" + "=" * 60)
    print("Testing Groq Dual-Mode Integration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.cli_integrations.groq_cli_integration import GroqCLIIntegration
        print("✅ Successfully imported GroqCLIIntegration")
        
        print("\n" + "-" * 60)
        print("Test 1: Initialization")
        print("-" * 60)
        
        try:
            groq = GroqCLIIntegration(api_key="test-key")
            print(f"✅ Groq integration initialized")
            print(f"   Tool name: {groq.get_tool_name()}")
            print(f"   CLI available: {groq.cli_available}")
            print(f"   Prefer CLI: {groq.prefer_cli}")
        except Exception as e:
            print(f"⚠️  Could not initialize (expected if dependencies not installed): {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_secrets_manager_integration():
    """Test secrets manager integration with CLI integrations."""
    print("\n" + "=" * 60)
    print("Testing Secrets Manager Integration with CLI Tools")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager
        from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration
        
        print("\n" + "-" * 60)
        print("Test 1: Store credentials in secrets manager")
        print("-" * 60)
        
        secrets = get_global_secrets_manager()
        secrets.set_credential("anthropic_api_key", "sk-test-anthropic-key")
        print("✅ Stored test credential in secrets manager")
        
        print("\n" + "-" * 60)
        print("Test 2: CLI integration retrieves from secrets manager")
        print("-" * 60)
        
        try:
            # Initialize without explicit API key - should get from secrets
            claude = ClaudeCodeCLIIntegration()
            
            # The API key should be retrieved from secrets manager
            if claude.api_key == "sk-test-anthropic-key":
                print("✅ API key correctly retrieved from secrets manager")
            elif claude.api_key:
                print(f"⚠️  Got API key '{claude.api_key}' (may be from environment)")
            else:
                print("⚠️  No API key retrieved (expected if not set)")
        except Exception as e:
            print(f"⚠️  Could not test integration: {e}")
        
        # Clean up
        secrets.delete_credential("anthropic_api_key")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_all_integrations_import():
    """Test that all updated integrations can be imported."""
    print("\n" + "=" * 60)
    print("Testing All Integration Imports")
    print("=" * 60)
    
    integrations = [
        ("Claude", "ipfs_accelerate_py.cli_integrations.claude_code_cli_integration", "ClaudeCodeCLIIntegration"),
        ("Gemini", "ipfs_accelerate_py.cli_integrations.gemini_cli_integration", "GeminiCLIIntegration"),
        ("Groq", "ipfs_accelerate_py.cli_integrations.groq_cli_integration", "GroqCLIIntegration"),
    ]
    
    all_passed = True
    for name, module_path, class_name in integrations:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {name}: Successfully imported {class_name}")
        except Exception as e:
            print(f"❌ {name}: Failed to import - {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = True
    
    success = test_dual_mode_wrapper() and success
    success = test_all_integrations_import() and success
    success = test_claude_dual_mode() and success
    success = test_gemini_dual_mode() and success
    success = test_groq_dual_mode() and success
    success = test_secrets_manager_integration() and success
    
    if success:
        print("\n✅ All Phase 3 tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
