#!/usr/bin/env python3
"""
Wrapper script to ensure GitHub CLI caching is used in VSCode tasks.

This script sets up the cache environment and then executes gh_api_cached.py
"""

import os
import sys
from pathlib import Path

# Set cache environment variables if not already set
os.environ.setdefault('CACHE_ENABLE_P2P', 'true')
os.environ.setdefault('CACHE_LISTEN_PORT', '9100')
os.environ.setdefault('CACHE_DEFAULT_TTL', '300')

# Set cache dir to workspace-local cache
workspace = Path(__file__).parent.parent
cache_dir = workspace / '.cache' / 'github_cli'
os.environ.setdefault('CACHE_DIR', str(cache_dir))

# Execute gh_api_cached.py with the provided arguments
tools_dir = workspace / 'tools'
gh_api_cached = tools_dir / 'gh_api_cached.py'

if not gh_api_cached.exists():
    print(f"Error: {gh_api_cached} not found", file=sys.stderr)
    sys.exit(1)

# Import and run the main function
sys.path.insert(0, str(tools_dir))
from gh_api_cached import main

sys.exit(main(sys.argv[1:]))
