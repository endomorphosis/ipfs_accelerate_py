"""
Configuration management for the refactored benchmark suite.
"""

from .benchmark_config import load_config_from_file, save_config_to_file

__all__ = ["load_config_from_file", "save_config_to_file"]