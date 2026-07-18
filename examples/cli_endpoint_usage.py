#!/usr/bin/env python3
"""
Example: Using CLI Endpoint Adapters with IPFS Accelerate

This example demonstrates how to use the CLI endpoint adapters to connect
to and multiplex CLI tools like Claude Code, OpenAI Codex, and Google Gemini
in the same way that we make calls to models using queues.
"""

import json
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def example_register_cli_endpoints():
    """Example: Register CLI endpoints"""
    print("=" * 70)
    print("Example 1: Registering CLI Endpoints")
    print("=" * 70)
    
    from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import (
        ClaudeCodeAdapter,
        OpenAICodexAdapter,
        GeminiCLIAdapter,
        register_cli_endpoint
    )
    
    # Register Claude Code CLI
    print("\n📝 Registering Claude Code CLI endpoint...")
    claude = ClaudeCodeAdapter(
        endpoint_id="claude_code_primary",
        config={
            "model": "claude-3-sonnet",
            "max_tokens": 4096,
            "temperature": 0.7
        }
    )
    result = register_cli_endpoint(claude)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    print(f"   Available: {result.get('available', False)}")
    
    # Register OpenAI CLI
    print("\n📝 Registering OpenAI CLI endpoint...")
    openai = OpenAICodexAdapter(
        endpoint_id="openai_codex_primary",
        config={
            "model": "gpt-4",
            "max_tokens": 2048
        }
    )
    result = register_cli_endpoint(openai)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    print(f"   Available: {result.get('available', False)}")
    
    # Register Google Gemini CLI
    print("\n📝 Registering Google Gemini CLI endpoint...")
    gemini = GeminiCLIAdapter(
        endpoint_id="gemini_pro_primary",
        config={
            "model": "gemini-pro",
            "temperature": 0.5
        }
    )
    result = register_cli_endpoint(gemini)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    print(f"   Available: {result.get('available', False)}")
    
    return True


def example_list_cli_endpoints():
    """Example: List all registered CLI endpoints"""
    print("\n" + "=" * 70)
    print("Example 2: Listing All CLI Endpoints")
    print("=" * 70)
    
    from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import list_cli_endpoints
    
    endpoints = list_cli_endpoints()
    
    print(f"\n📊 Total CLI Endpoints: {len(endpoints)}")
    for endpoint in endpoints:
        print(f"\n  • {endpoint['endpoint_id']}")
        print(f"    Type: {endpoint['endpoint_type']}")
        print(f"    CLI Path: {endpoint.get('cli_path', 'N/A')}")
        print(f"    Available: {endpoint.get('available', False)}")
        print(f"    Requests: {endpoint['stats']['requests']}")
        print(f"    Success Rate: {endpoint['stats']['successes']}/{endpoint['stats']['requests']}")
    
    return True


def example_cli_inference():
    """Example: Run inference using CLI endpoints"""
    print("\n" + "=" * 70)
    print("Example 3: Running Inference with CLI Endpoints")
    print("=" * 70)
    
    from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import execute_cli_inference
    
    prompt = "What are the key benefits of using IPFS for distributed model storage?"
    
    print(f"\n🔍 Prompt: {prompt}")
    
    # Try Claude (if available)
    print("\n🤖 Attempting inference with Claude Code CLI...")
    result = execute_cli_inference(
        endpoint_id="claude_code_primary",
        prompt=prompt,
        task_type="text_generation",
        timeout=30
    )
    
    if result.get("status") == "success":
        print(f"   ✅ Success!")
        print(f"   Response: {result.get('result', 'N/A')[:200]}...")
        print(f"   Elapsed: {result.get('elapsed_time', 0):.2f}s")
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Try OpenAI (if available)
    print("\n🤖 Attempting inference with OpenAI CLI...")
    result = execute_cli_inference(
        endpoint_id="openai_codex_primary",
        prompt=prompt,
        task_type="text_generation",
        timeout=30
    )
    
    if result.get("status") == "success":
        print(f"   ✅ Success!")
        print(f"   Response: {result.get('result', 'N/A')[:200]}...")
        print(f"   Elapsed: {result.get('elapsed_time', 0):.2f}s")
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return True


def example_multiplexed_inference():
    """Example: Use multiplexed inference with CLI fallbacks"""
    print("\n" + "=" * 70)
    print("Example 4: Multiplexed Inference with CLI Fallbacks")
    print("=" * 70)
    
    try:
        from ipfs_accelerate_py.mcp_server.tools.enhanced_inference import (
            HAVE_CLI_ADAPTERS,
            CLI_PROVIDERS
        )
        
        if not HAVE_CLI_ADAPTERS:
            print("\n⚠️  CLI Adapters not available in enhanced_inference")
            return False
        
        print(f"\n✅ CLI Adapters are integrated!")
        print(f"\n📋 Available CLI Providers:")
        for provider, config in CLI_PROVIDERS.items():
            print(f"   • {provider}: {config['description']}")
            print(f"     Models: {', '.join(config['models'])}")
        
        print("\n💡 Usage in multiplexed inference:")
        print("   You can now specify CLI providers in model preferences:")
        print("   - 'claude_cli/claude-3-sonnet'")
        print("   - 'openai_cli/gpt-4'")
        print("   - 'gemini_cli/gemini-pro'")
        
        print("\n   Example:")
        print("   multiplex_inference(")
        print("       prompt='Your prompt here',")
        print("       model_preferences=[")
        print("           'claude_cli/claude-3-sonnet',")
        print("           'openai_cli/gpt-4',")
        print("           'openai/gpt-3.5-turbo',  # API fallback")
        print("       ]")
        print("   )")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Could not import enhanced_inference: {e}")
        return False


def example_queue_monitoring():
    """Example: Monitor queue status with CLI endpoints"""
    print("\n" + "=" * 70)
    print("Example 5: Queue Monitoring with CLI Endpoints")
    print("=" * 70)
    
    try:
        from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import CLI_ADAPTER_REGISTRY
        
        print(f"\n📊 CLI Endpoint Registry Status:")
        print(f"   Total Registered: {len(CLI_ADAPTER_REGISTRY)}")
        
        for endpoint_id, adapter in CLI_ADAPTER_REGISTRY.items():
            stats = adapter.get_stats()
            print(f"\n   • {endpoint_id}:")
            print(f"     Available: {stats['available']}")
            print(f"     CLI Path: {stats['cli_path']}")
            print(f"     Total Requests: {stats['stats']['requests']}")
            print(f"     Success Rate: {stats['stats']['successes']}/{stats['stats']['requests']}")
            print(f"     Avg Time: {stats['stats']['avg_time']:.2f}s")
        
        print("\n💡 These endpoints are now part of the queue monitoring system!")
        print("   They appear in get_queue_status() alongside other endpoints.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Run all examples"""
    print("\n" + "🚀" * 35)
    print("CLI Endpoint Adapters - Usage Examples")
    print("🚀" * 35)
    
    try:
        # Example 1: Register endpoints
        example_register_cli_endpoints()
        
        # Example 2: List endpoints
        example_list_cli_endpoints()
        
        # Example 3: Run inference
        example_cli_inference()
        
        # Example 4: Multiplexed inference
        example_multiplexed_inference()
        
        # Example 5: Queue monitoring
        example_queue_monitoring()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed!")
        print("=" * 70)
        
        print("\n📚 Next Steps:")
        print("   1. Install the CLI tools (claude, openai, gemini/gcloud)")
        print("   2. Configure API keys and authentication")
        print("   3. Test with real inference requests")
        print("   4. Integrate into your MCP server workflow")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
