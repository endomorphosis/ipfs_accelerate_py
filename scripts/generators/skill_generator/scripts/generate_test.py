#!/usr/bin/env python3
"""
Generate Test Script

This script provides a command-line interface for generating test files for HuggingFace models.
"""

import sys
import os
import logging

# Make parent directory available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator_core.cli import GeneratorCLI

def main():
    """Main entry point."""
    cli = GeneratorCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())