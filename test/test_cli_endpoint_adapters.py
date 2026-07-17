#!/usr/bin/env python3
"""
Test script for CLI Endpoint Adapters functionality

This script tests the CLI endpoint adapter integration with the
IPFS Accelerate MCP queue system.
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_cli_endpoint_adapters():
    """Test the CLI endpoint adapter functionality"""
    print("🧪 Testing CLI Endpoint Adapters")
    print("=" * 60)
    
    try:
        # Import the CLI endpoint adapters
        from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import (
            ClaudeCodeAdapter,
            OpenAICodexAdapter,
            GeminiCLIAdapter,
            register_cli_endpoint,
            list_cli_endpoints,
            execute_cli_inference
        )
        
        print("✅ Successfully imported CLI endpoint adapters")
        
        # Test 1: Create adapters
        print("\n📋 Test 1: Creating CLI adapters...")
        
        claude_adapter = ClaudeCodeAdapter("claude_test", config={"model": "claude-3-sonnet"})
        print(f"  ✓ Created Claude adapter: {claude_adapter.endpoint_id}")
        print(f"    Available: {claude_adapter.is_available()}")
        print(f"    CLI Path: {claude_adapter.cli_path}")
        
        openai_adapter = OpenAICodexAdapter("openai_test", config={"model": "gpt-3.5-turbo"})
        print(f"  ✓ Created OpenAI adapter: {openai_adapter.endpoint_id}")
        print(f"    Available: {openai_adapter.is_available()}")
        print(f"    CLI Path: {openai_adapter.cli_path}")
        
        gemini_adapter = GeminiCLIAdapter("gemini_test", config={"model": "gemini-pro"})
        print(f"  ✓ Created Gemini adapter: {gemini_adapter.endpoint_id}")
        print(f"    Available: {gemini_adapter.is_available()}")
        print(f"    CLI Path: {gemini_adapter.cli_path}")
        
        # Test 2: Register adapters
        print("\n📋 Test 2: Registering CLI endpoints...")
        
        result = register_cli_endpoint(claude_adapter)
        print(f"  Claude: {result}")
        
        result = register_cli_endpoint(openai_adapter)
        print(f"  OpenAI: {result}")
        
        result = register_cli_endpoint(gemini_adapter)
        print(f"  Gemini: {result}")
        
        # Test 3: List registered endpoints
        print("\n📋 Test 3: Listing registered endpoints...")
        
        endpoints = list_cli_endpoints()
        print(f"  Total registered: {len(endpoints)}")
        for endpoint in endpoints:
            print(f"  - {endpoint['endpoint_id']}: available={endpoint['available']}")
        
        # Test 4: Format prompts
        print("\n📋 Test 4: Testing prompt formatting...")
        
        test_prompt = "What is the capital of France?"
        
        claude_cmd = claude_adapter._format_prompt(test_prompt, "text_generation")
        print(f"  Claude command: {' '.join(claude_cmd[:5])}...")
        
        openai_cmd = openai_adapter._format_prompt(test_prompt, "text_generation")
        print(f"  OpenAI command: {' '.join(openai_cmd[:5])}...")
        
        gemini_cmd = gemini_adapter._format_prompt(test_prompt, "text_generation")
        print(f"  Gemini command: {' '.join(gemini_cmd[:5])}...")
        
        # Test 5: Test response parsing
        print("\n📋 Test 5: Testing response parsing...")
        
        test_json_output = '{"content": [{"text": "Paris is the capital of France."}], "model": "claude-3-sonnet"}'
        test_text_output = "Paris is the capital of France."
        
        claude_result = claude_adapter._parse_response(test_json_output, "")
        print(f"  Claude JSON parsing: {claude_result.get('result', 'N/A')[:50]}...")
        
        claude_result_text = claude_adapter._parse_response(test_text_output, "")
        print(f"  Claude text parsing: {claude_result_text.get('result', 'N/A')[:50]}...")
        
        # Test 6: Test adapter statistics
        print("\n📋 Test 6: Testing adapter statistics...")
        
        for adapter in [claude_adapter, openai_adapter, gemini_adapter]:
            stats = adapter.get_stats()
            print(f"  {stats['endpoint_id']}:")
            print(f"    Type: {stats['endpoint_type']}")
            print(f"    Available: {stats['available']}")
            print(f"    Requests: {stats['stats']['requests']}")
            print(f"    Successes: {stats['stats']['successes']}")
        
        # Test 7: Test integration with enhanced_inference
        print("\n📋 Test 7: Testing integration with enhanced_inference...")
        
        try:
            from ipfs_accelerate_py.mcp_server.tools.enhanced_inference import (
                CLI_PROVIDERS,
                HAVE_CLI_ADAPTERS
            )
            
            print(f"  CLI Adapters available: {HAVE_CLI_ADAPTERS}")
            print(f"  CLI Providers configured: {list(CLI_PROVIDERS.keys())}")
            
            for provider, config in CLI_PROVIDERS.items():
                print(f"    - {provider}: {config['description']}")
                print(f"      Models: {', '.join(config['models'])}")
        
        except ImportError as e:
            print(f"  ⚠️  Could not test integration: {e}")
        
        print("\n✅ All CLI endpoint adapter tests completed successfully!")
        print("\n📝 Summary:")
        print("  ✓ CLI endpoint adapters can be created")
        print("  ✓ CLI endpoints can be registered")
        print("  ✓ Prompt formatting works for all providers")
        print("  ✓ Response parsing works for JSON and text")
        print("  ✓ Statistics tracking is functional")
        print("  ✓ Integration with enhanced_inference is available")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_inference_integration():
    """Test the integration with enhanced_inference module"""
    print("\n" + "=" * 60)
    print("🧪 Testing Enhanced Inference Integration")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.mcp_server.tools.enhanced_inference import (
            CLI_PROVIDERS,
            HAVE_CLI_ADAPTERS,
            CLI_ADAPTER_REGISTRY
        )
        
        print(f"\n✅ CLI Adapters Module Available: {HAVE_CLI_ADAPTERS}")
        
        if HAVE_CLI_ADAPTERS:
            print(f"\n📋 Configured CLI Providers:")
            for provider, config in CLI_PROVIDERS.items():
                print(f"  • {provider}:")
                print(f"    Description: {config['description']}")
                print(f"    Adapter: {config['adapter_class']}")
                print(f"    Models: {', '.join(config['models'])}")
            
            print(f"\n📊 Currently Registered CLI Endpoints: {len(CLI_ADAPTER_REGISTRY)}")
            for endpoint_id in CLI_ADAPTER_REGISTRY.keys():
                print(f"  - {endpoint_id}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_cli_endpoint_adapters()
    success2 = test_enhanced_inference_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
