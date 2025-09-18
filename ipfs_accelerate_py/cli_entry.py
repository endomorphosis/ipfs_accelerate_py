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
        # Add the current directory and parent directory to the path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        root_dir = os.path.dirname(parent_dir)
        
        # Add paths for different installation scenarios
        for path in [parent_dir, root_dir, current_dir]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Try importing the CLI module
        try:
            from cli import main as cli_main
            return cli_main()
        except ImportError:
            try:
                # Try importing from the package
                from ipfs_accelerate_py.cli import main as cli_main  
                return cli_main()
            except ImportError:
                try:
                    # Try absolute import with full path
                    import importlib.util
                    cli_path = os.path.join(root_dir, 'cli.py')
                    if os.path.exists(cli_path):
                        spec = importlib.util.spec_from_file_location("cli", cli_path)
                        cli_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(cli_module)
                        return cli_module.main()
                    else:
                        raise ImportError(f"CLI module not found at {cli_path}")
                except Exception as e:
                    print(f"Error: Could not import CLI module: {e}", file=sys.stderr)
                    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
                    print(f"Script location: {current_dir}", file=sys.stderr)
                    print(f"Python path: {sys.path[:5]}", file=sys.stderr)
                    print("", file=sys.stderr)
                    print("Possible solutions:", file=sys.stderr)
                    print("1. Run directly: python cli.py mcp start", file=sys.stderr)
                    print("2. Run as module: python -m ipfs_accelerate_py.cli_entry mcp start", file=sys.stderr) 
                    print("3. Install package: pip install -e .", file=sys.stderr)
                    return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())