#!/usr/bin/env python3
"""
Example: Auto-Healing Error Handler

This example demonstrates the automatic error handling and GitHub integration
features of IPFS Accelerate CLI.

Features demonstrated:
1. Automatic error capture with stack traces
2. GitHub issue creation for errors
3. Draft PR generation
4. GitHub Copilot auto-healing
"""

import sys
import os
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py.error_handler import CLIErrorHandler


def simulate_cli_errors():
    """Simulate various CLI errors to demonstrate auto-healing."""
    
    print("=" * 80)
    print("Auto-Healing Error Handler Example")
    print("=" * 80)
    print()
    
    # Example 1: Basic error capture (no auto-features)
    print("Example 1: Basic Error Capture (no auto-features)")
    print("-" * 80)
    
    handler = CLIErrorHandler(
        repo='endomorphosis/ipfs_accelerate_py',
        enable_auto_issue=False,
        enable_auto_pr=False,
        enable_auto_heal=False
    )
    
    try:
        # Simulate an error
        raise ValueError("Invalid model configuration: missing 'model_path' parameter")
    except Exception as e:
        signature = handler.capture_error(
            e,
            context={
                'command': 'inference generate',
                'model': 'bert-base-uncased',
                'operation': 'load_model'
            }
        )
        print(f"✓ Error captured with signature: {signature[:16] if signature else 'N/A'}...")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Severity: {handler._determine_severity(e)}")
        print(f"  Captured errors: {len(handler._captured_errors)}")
    
    print()
    
    # Example 2: Error capture with auto-issue enabled (simulation)
    print("Example 2: Auto-Issue Creation (simulated)")
    print("-" * 80)
    
    handler_auto = CLIErrorHandler(
        repo='endomorphosis/ipfs_accelerate_py',
        enable_auto_issue=True,
        enable_auto_pr=False,
        enable_auto_heal=False
    )
    
    try:
        # Simulate a different error
        raise RuntimeError("CUDA out of memory: tried to allocate 2.50 GiB")
    except Exception as e:
        # Capture the error
        handler_auto.capture_error(e, context={'gpu': 'NVIDIA RTX 3090', 'batch_size': 32})
        print(f"✓ Error captured")
        print(f"  Auto-issue creation: {'ENABLED' if handler_auto.enable_auto_issue else 'DISABLED'}")
        print(f"  Note: Actual issue creation requires GitHub CLI authentication")
        print(f"  To test: export IPFS_AUTO_ISSUE=true && gh auth login")
    
    print()
    
    # Example 3: Full auto-healing pipeline (simulation)
    print("Example 3: Full Auto-Healing Pipeline (simulated)")
    print("-" * 80)
    
    handler_full = CLIErrorHandler(
        repo='endomorphosis/ipfs_accelerate_py',
        enable_auto_issue=True,
        enable_auto_pr=True,
        enable_auto_heal=True
    )
    
    try:
        # Simulate a critical error
        raise MemoryError("Cannot allocate memory for tensor of shape [1024, 1024, 1024]")
    except Exception as e:
        handler_full.capture_error(e)
        print(f"✓ Error captured with full auto-healing pipeline")
        print(f"  Auto-issue: {'ENABLED' if handler_full.enable_auto_issue else 'DISABLED'}")
        print(f"  Auto-PR: {'ENABLED' if handler_full.enable_auto_pr else 'DISABLED'}")
        print(f"  Auto-heal: {'ENABLED' if handler_full.enable_auto_heal else 'DISABLED'}")
        print(f"  Severity: {handler_full._determine_severity(e)}")
        print()
        print("  When enabled, this would:")
        print("    1. Create a GitHub issue with full error details")
        print("    2. Generate a draft PR to fix the issue")
        print("    3. Invoke GitHub Copilot to suggest fixes")
    
    print()
    
    # Example 4: CLI function wrapping
    print("Example 4: CLI Function Wrapping")
    print("-" * 80)
    
    handler_wrapper = CLIErrorHandler(
        repo='endomorphosis/ipfs_accelerate_py',
        enable_auto_issue=False
    )
    
    @handler_wrapper.wrap_cli_main
    def example_cli_function():
        """Example CLI function that might fail."""
        print("  Executing CLI function...")
        # Simulate some work
        print("  Processing model inference...")
        # This would normally raise an error
        # raise ValueError("Example error")
        print("  ✓ CLI function completed successfully")
        return 0
    
    result = example_cli_function()
    print(f"  Exit code: {result}")
    
    print()
    
    # Example 5: Environment-based configuration
    print("Example 5: Environment-Based Configuration")
    print("-" * 80)
    
    print("  Current environment settings:")
    print(f"    IPFS_AUTO_ISSUE: {os.environ.get('IPFS_AUTO_ISSUE', 'not set')}")
    print(f"    IPFS_AUTO_PR: {os.environ.get('IPFS_AUTO_PR', 'not set')}")
    print(f"    IPFS_AUTO_HEAL: {os.environ.get('IPFS_AUTO_HEAL', 'not set')}")
    print(f"    IPFS_REPO: {os.environ.get('IPFS_REPO', 'not set (default: endomorphosis/ipfs_accelerate_py)')}")
    print()
    print("  To enable auto-healing features:")
    print("    export IPFS_AUTO_ISSUE=true")
    print("    export IPFS_AUTO_PR=true")
    print("    export IPFS_AUTO_HEAL=true")
    print()
    print("  Then run:")
    print("    ipfs-accelerate <any-command>")
    
    print()
    
    # Cleanup
    handler.cleanup()
    handler_auto.cleanup()
    handler_full.cleanup()
    handler_wrapper.cleanup()
    
    print("=" * 80)
    print("Example Complete")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • Error capture works without GitHub integration")
    print("  • Auto-issue requires GitHub CLI authentication (gh auth login)")
    print("  • Auto-PR requires auto-issue to be enabled")
    print("  • Auto-heal requires Copilot SDK (pip install github-copilot-sdk)")
    print()
    print("For more information, see:")
    print("  docs/AUTO_HEALING_CONFIGURATION.md")
    print()


def demonstrate_error_aggregation():
    """Demonstrate P2P error aggregation features."""
    
    print()
    print("=" * 80)
    print("Error Aggregation Example")
    print("=" * 80)
    print()
    
    print("Error aggregation features:")
    print("  • Shares errors across distributed instances")
    print("  • Deduplicates similar errors")
    print("  • Bundles multiple occurrences before creating issues")
    print("  • Prevents duplicate GitHub issues")
    print()
    print("Configuration:")
    print("  • Bundle interval: 15 minutes (default)")
    print("  • Minimum occurrences: 3 errors (default)")
    print("  • Deduplication: Based on error signature")
    print()
    print("Note: Error aggregation requires P2P connectivity")
    print("      and is automatically used when available.")
    print()


if __name__ == '__main__':
    try:
        simulate_cli_errors()
        demonstrate_error_aggregation()
        print("✓ Example completed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n✗ Example interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
