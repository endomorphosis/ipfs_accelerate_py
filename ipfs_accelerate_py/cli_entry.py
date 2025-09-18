#!/usr/bin/env python3
"""
IPFS Accelerate CLI Entry Point

This module provides the main entry point for the ipfs-accelerate command line tool.
"""

import sys
import os

def main():
    """Main entry point for the ipfs-accelerate command."""
    try:
        # Try to import from the parent directory's cli module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        from cli import main as cli_main
        return cli_main()
    except ImportError as e:
        # Fallback: try different import paths
        try:
            # Try importing as if installed as package
            from ipfs_accelerate_py.cli import main as cli_main
            return cli_main()
        except ImportError:
            try:
                # Try direct import from current directory
                import cli
                return cli.main()
            except ImportError:
                print(f"Error: Could not import CLI module: {e}", file=sys.stderr)
                print("Please ensure ipfs_accelerate_py is properly installed.", file=sys.stderr)
                print("You can also run directly with: python cli.py", file=sys.stderr)
                return 1

if __name__ == '__main__':
    sys.exit(main())