#!/usr/bin/env python3
"""
Run the browser recovery strategy demo.

This script provides a convenient way to demonstrate browser recovery strategies
with different browsers and model types.

Usage:
    python distributed_testing/run_browser_recovery_demo.py
    python distributed_testing/run_browser_recovery_demo.py --browser firefox --model whisper-tiny
"""

import os
import sys
import argparse
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("browser_recovery_demo")

def get_script_dir():
    """Get the directory of this script."""
    return os.path.dirname(os.path.abspath(__file__))

def setup_environment():
    """Setup the environment for the demo."""
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(get_script_dir())
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

async def run_demo():
    """Run the browser recovery demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Browser Recovery Strategy Demo")
    parser.add_argument("--browser", type=str, default="chrome", 
                      help="Browser to use (chrome, firefox, edge)")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                      help="Model to test")
    parser.add_argument("--platform", type=str, default="webgpu", choices=["webgpu", "webnn"], 
                      help="Platform to use")
    parser.add_argument("--no-failures", action="store_true", 
                      help="Don't inject artificial failures")
    parser.add_argument("--no-stats", action="store_true", 
                      help="Don't show recovery statistics")
    
    args = parser.parse_args()
    
    # Import the demo module
    from integration_examples.browser_recovery_integration import BrowserRecoveryDemo
    
    # Create and run the demo
    demo = BrowserRecoveryDemo(
        browser_name=args.browser,
        model_name=args.model,
        platform=args.platform,
        inject_failures=not args.no_failures,
        show_statistics=not args.no_stats
    )
    
    await demo.run_with_fault_tolerance()

def main():
    """Main entry point."""
    setup_environment()
    
    print("=" * 80)
    print("Browser Recovery Strategy Demo")
    print("=" * 80)
    print("This demo shows how to integrate advanced browser recovery strategies")
    print("with the circuit breaker pattern for browser automation.")
    print()
    print("For comprehensive documentation, see:")
    print("  - ADVANCED_BROWSER_RECOVERY_STRATEGIES.md")
    print("  - ADVANCED_FAULT_TOLERANCE_BROWSER_INTEGRATION.md")
    print()
    print("The demo will run with various browser failures and show how they are")
    print("automatically recovered using model-aware, browser-specific strategies.")
    print("=" * 80)
    print()
    
    # Run the demo
    asyncio.run(run_demo())

if __name__ == "__main__":
    main()