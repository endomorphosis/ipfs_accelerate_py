#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line script to run the refactored generator.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import generator components
from generator_core import ConfigManager, ComponentRegistry, GeneratorCore
from generator_core.cli import main as cli_main

if __name__ == "__main__":
    # Run the CLI main function
    sys.exit(cli_main())