#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script for running the refactored benchmark suite.

This script provides a command-line interface to run the refactored benchmark suite.
It handles the required environment setup and command-line arguments.
"""

import os
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run HuggingFace Model Benchmarks")
    
    # Add benchmark suite path
    parser.add_argument(
        "--suite-path", type=str, default=os.path.join(SCRIPT_DIR, "refactored_benchmark_suite"),
        help="Path to the refactored benchmark suite directory"
    )
    
    # Pass all remaining arguments to the benchmark suite
    parser.add_argument(
        "benchmark_args", nargs=argparse.REMAINDER,
        help="Arguments to pass to the benchmark suite"
    )
    
    args = parser.parse_args()
    
    # Ensure the benchmark suite path exists
    suite_path = Path(args.suite_path)
    if not suite_path.exists() or not suite_path.is_dir():
        print(f"Error: Benchmark suite path does not exist: {suite_path}")
        return 1
    
    # Add the parent directory to Python path to ensure imports work
    parent_dir = suite_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Import and run the benchmark suite
    try:
        from refactored_benchmark_suite.__main__ import main as benchmark_main
        
        # If no benchmark args provided, show help
        if not args.benchmark_args:
            args.benchmark_args = ["--help"]
        
        # Set sys.argv for the benchmark suite
        orig_argv = sys.argv
        sys.argv = [sys.argv[0]] + args.benchmark_args
        
        # Run the benchmark suite
        result = benchmark_main()
        
        # Restore sys.argv
        sys.argv = orig_argv
        
        return result
        
    except ImportError as e:
        print(f"Error importing benchmark suite: {e}")
        print(f"Please ensure the benchmark suite is correctly installed at {suite_path}")
        return 1
    except Exception as e:
        print(f"Error running benchmark suite: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())