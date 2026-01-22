#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration management for the refactored generator suite.
Handles loading, accessing, and validating configuration.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class ConfigManager:
    """Configuration manager for the generator system."""

    DEFAULT_CONFIG_PATHS = [
        # Current directory config
        "./config.yaml",
        "./config.json",
        
        # User home directory config
        "~/.config/hf_generator/config.yaml",
        "~/.config/hf_generator/config.json",
        
        # Installation directory config
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.json")
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a configuration file. If not provided,
                         default locations will be searched.
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Set default values for missing entries
        self._set_defaults()
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from specified path or default locations.
        
        Args:
            config_path: Optional path to a configuration file.
            
        Returns:
            Dict containing the loaded configuration.
        """
        config = {}
        
        # If config_path is provided, only try to load from that path
        if config_path:
            if not os.path.exists(config_path):
                self.logger.warning(f"Configuration file not found: {config_path}")
                return {}
                
            config = self._load_config_file(config_path)
            if config:
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                self.logger.warning(f"Failed to load configuration from {config_path}")
                return {}
                
        # Otherwise, search default paths
        for path in self.DEFAULT_CONFIG_PATHS:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                config = self._load_config_file(expanded_path)
                if config:
                    self.logger.info(f"Loaded configuration from {expanded_path}")
                    break
        
        return config
    
    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            path: Path to the configuration file.
            
        Returns:
            Dict containing the loaded configuration, or an empty dict if loading failed.
        """
        try:
            if path.endswith(('.yaml', '.yml')):
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Unsupported configuration file format: {path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration from {path}: {e}")
            return {}
    
    def _set_defaults(self) -> None:
        """Set default values for missing configuration entries."""
        # Default output directory
        if 'output_dir' not in self.config:
            self.config['output_dir'] = './generated_tests'
        
        # Default templates directory
        if 'templates_dir' not in self.config:
            self.config['templates_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
        
        # Default model configurations
        if 'models' not in self.config:
            self.config['models'] = {}
        
        # Default hardware options
        if 'hardware' not in self.config:
            self.config['hardware'] = {
                'prefer_cuda': True,
                'prefer_rocm': False,
                'prefer_mps': False,
                'prefer_openvino': False,
                'prefer_webnn': False,
                'prefer_webgpu': False
            }
        
        # Default dependency options
        if 'dependencies' not in self.config:
            self.config['dependencies'] = {
                'use_mocks': False,
                'ignore_missing': False
            }
        
        # Default syntax options
        if 'syntax' not in self.config:
            self.config['syntax'] = {
                'auto_fix': True,
                'strict_validation': False
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: The configuration key to get.
            default: Default value to return if the key is not found.
            
        Returns:
            The configuration value, or default if the key is not found.
        """
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key to set.
            value: The value to set.
        """
        parts = key.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def merge(self, additional_config: Dict[str, Any]) -> None:
        """Merge additional configuration values.
        
        Args:
            additional_config: Additional configuration to merge.
        """
        self._merge_dict(self.config, additional_config)
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge dictionaries.
        
        Args:
            target: Target dictionary to merge into.
            source: Source dictionary to merge from.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary.
        
        Returns:
            Dict containing the entire configuration.
        """
        return self.config.copy()
    
    def save(self, path: str) -> bool:
        """Save the configuration to a file.
        
        Args:
            path: Path to save the configuration to.
            
        Returns:
            True if the save was successful, False otherwise.
        """
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            if path.endswith(('.yaml', '.yml')):
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif path.endswith('.json'):
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                self.logger.warning(f"Unsupported configuration file format: {path}")
                return False
                
            self.logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration to {path}: {e}")
            return False


# Global instance for simpler access
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global config manager instance.
    
    Returns:
        The global ConfigManager instance.
    """
    return config_manager