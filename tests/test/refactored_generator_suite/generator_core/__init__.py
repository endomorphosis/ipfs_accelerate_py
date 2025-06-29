#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generator core package for the refactored generator suite.
Provides access to core generator components.
"""

import logging

from .config import ConfigManager, get_config
from .registry import ComponentRegistry
from .generator import GeneratorCore
from .cli import main as cli_main

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Package exports
__all__ = [
    'ConfigManager',
    'get_config',
    'ComponentRegistry',
    'GeneratorCore',
    'cli_main'
]